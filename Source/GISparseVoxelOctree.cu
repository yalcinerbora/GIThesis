#include "GISparseVoxelOctree.h"
#include <cuda_gl_interop.h>
#include "GICudaAllocator.h"
#include "GIKernels.cuh"
#include "CudaTimer.h"
#include "Macros.h"

__global__ void AllocLocInit(unsigned int* dSVOEmptyLoc, 
							 const unsigned int maxCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= maxCount) return;
	dSVOEmptyLoc[globalId] = (maxCount - globalId - 1) * 8;
}

GISparseVoxelOctree::GISparseVoxelOctree(GLuint lightIntensityTex)
	: dSVO()
	, lightIntensityTexLink(nullptr)
	, dSVOLocIndex(1)
	, dSVOConstants(1)
	, tSVODense(0)
	, vaoNormPosData(512)
	, vaoColorData(512)
{
	CUDA_CHECK(cudaGraphicsGLRegisterImage(&lightIntensityTexLink, lightIntensityTex,
											GL_TEXTURE_2D,
											cudaGraphicsMapFlagsWriteDiscard));
}

GISparseVoxelOctree::~GISparseVoxelOctree()
{
	if(lightIntensityTexLink) CUDA_CHECK(cudaGraphicsUnregisterResource(lightIntensityTexLink));
	if(lightBufferLink) CUDA_CHECK(cudaGraphicsUnregisterResource(lightBufferLink));
	if(shadowMapArrayTexLink) CUDA_CHECK(cudaGraphicsUnregisterResource(shadowMapArrayTexLink));
	if(tSVODense) CUDA_CHECK(cudaDestroyTextureObject(tSVODense));
	if(denseArray) CUDA_CHECK(cudaFreeArray(denseArray));
}

__global__ void fastreadkernel3D(cudaTextureObject_t texture,
							   unsigned int* gOut)
{
	uint3 globalId;
	globalId.x = threadIdx.x + blockIdx.x * blockDim.x;
	globalId.y = threadIdx.y + blockIdx.y * blockDim.y;
	globalId.z = threadIdx.z + blockIdx.z * blockDim.z;

	float3 texUV;
	texUV.x = static_cast<float>(globalId.x);
	texUV.y = static_cast<float>(globalId.y);
	texUV.z = static_cast<float>(globalId.z);

	unsigned int currentNode = tex3D<unsigned int>(texture, texUV.x, texUV.y, texUV.z);
	gOut[globalId.z * GI_DENSE_SIZE * GI_DENSE_SIZE +
		 globalId.y	* GI_DENSE_SIZE + 
		 globalId.x] = currentNode;
}

__global__ void fastreadkernel(cudaTextureObject_t texture,
								 unsigned int* gOut)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	
	float3 texUV;
	texUV.x = static_cast<float>(globalId % GI_DENSE_SIZE);
	texUV.y = static_cast<float>((globalId / GI_DENSE_SIZE) % GI_DENSE_SIZE);
	texUV.z = static_cast<float>(globalId / GI_DENSE_SIZE / GI_DENSE_SIZE);

	/*x = idx % (max_x)
		idx /= (max_x)
		y = idx % (max_y)
		idx /= (max_y)
		z = idx
		return (x, y, z)*/

	unsigned int currentNode = tex3D<unsigned int>(texture, texUV.x, texUV.y, texUV.z);
	printf("%d TexCoordXYZ %f, %f, %f\n", globalId, texUV.x, texUV.y, texUV.z);

	gOut[globalId] = currentNode;
}

void GISparseVoxelOctree::LinkAllocators(GICudaAllocator** newAllocators,
										 size_t allocatorSize)
{
	allocatorGrids.clear();
	allocators.resize(allocatorSize);
	allocatorGrids.resize(allocatorSize);

	assert(allocatorSize > 0);
	assert(newAllocators != nullptr);

	std::copy(newAllocators, newAllocators + allocatorSize, allocators.data());
	for(unsigned int i = 0; i < allocatorSize; i++)
		allocatorGrids[i] = newAllocators[i]->GetVoxelGridHost();

	// TODO: More Dynamic Allocation Scheme
	size_t totalAlloc = GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE;
	float multiplier = static_cast<float>(1.0 - std::pow(0.220f, (allocatorGrids[0].depth - GI_DENSE_LEVEL))) / (1.0f - 0.220f);
	
	for(unsigned int i = 0; i < allocatorSize; i++)
	{
		float depthMultiplier = 1;
		if(i == 0) depthMultiplier = multiplier;
		totalAlloc += static_cast<unsigned int>(allocators[i]->NumPages() * GI_PAGE_SIZE * depthMultiplier);
	}
	dSVO.Resize(totalAlloc);
	dSVOLock.Resize(totalAlloc);

	// Allocation Location Stack
	totalEightNodes = static_cast<unsigned int>((totalAlloc - GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE) / 8);
	dSVOEmptyLoc.Resize(totalEightNodes);
	
	totalAlloc += static_cast<unsigned int>((1.0 - std::pow(8.0f, GI_DENSE_LEVEL)) / (1.0f - 8.0f));
	dSVOMaterial.Resize(totalAlloc);
	
	dSVODense = dSVO.Data();
	dSVOSparse = dSVO.Data() + (GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);

	// Constants Set
	unsigned int totalLevel = static_cast<unsigned int>(allocatorGrids[0].depth + allocatorSize - 1);
	hSVOConstants.denseDepth = GI_DENSE_LEVEL;
	hSVOConstants.denseDim = GI_DENSE_SIZE;
	hSVOConstants.totalDepth = totalLevel;
	hSVOConstants.numCascades = static_cast<unsigned int>(allocatorSize);
	CUDA_CHECK(cudaMemcpy(dSVOConstants.Data(), 
						  &hSVOConstants, 
						  sizeof(CSVOConstants), 
						  cudaMemcpyHostToDevice));

	// Cuda Array that will be used to map dense part to 3d texture
	cudaChannelFormatDesc fd = cudaCreateChannelDesc<unsigned int>();
	if(denseArray) CUDA_CHECK(cudaFreeArray(denseArray));
	CUDA_CHECK(cudaMalloc3DArray(&denseArray,
								 &fd,
								 {GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE},
								 cudaArrayDefault));

	// Texture of SVO Dense
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = denseArray;
	
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	if(tSVODense != 0) CUDA_CHECK(cudaDestroyTextureObject(tSVODense));
	CUDA_CHECK(cudaCreateTextureObject(&tSVODense, &resDesc, &texDesc, nullptr));
}

double GISparseVoxelOctree::UpdateSVO()
{
	CudaTimer timer;
	timer.Start();

	// Reset EmptyLocationArrays
	// Reset Nodes
	unsigned int usedNodeCount;
	CUDA_CHECK(cudaMemcpy(&usedNodeCount, dSVOLocIndex.Data(), sizeof(unsigned int),
						  cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(dSVOLocIndex.Data(), &totalEightNodes, sizeof(unsigned int),
						  cudaMemcpyHostToDevice));
	dSVO.Memset(0xFF, 0, dSVO.Size());
	dSVOLock.Memset(0xFF, 0, dSVOLock.Size());

	uint32_t gridSize = (totalEightNodes + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	AllocLocInit<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVOEmptyLoc.Data(),
		totalEightNodes
	);
	CUDA_KERNEL_CHECK();
	
	// Init Done
	// Call SVOReconstruct for each allocator
	for(unsigned int i = 0; i < 1/*allocators.size()*/; i++)
	{
		gridSize = (allocators[i]->NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) / 
					GI_THREAD_PER_BLOCK;
		SVOReconstruct<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(
			dSVOSparse,
			dSVODense,
			dSVOLock.Data(),

			dSVOEmptyLoc.Data(),
			*dSVOLocIndex.Data(),
			totalEightNodes,
			
			// SVO Alloc Location Holding Data
			allocators[i]->GetVoxelPagesDevice(),

			i,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}

	// Construction Complete Copy Dense to Array
	cudaMemcpy3DParms params = {0};
	params.dstArray = denseArray;
	params.srcPtr = 
	{
		dSVODense, 
		GI_DENSE_SIZE * sizeof(unsigned int), 
		GI_DENSE_SIZE, 
		GI_DENSE_SIZE
	};
	params.extent = {GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE};
	params.kind = cudaMemcpyDeviceToDevice;
	CUDA_CHECK(cudaMemcpy3D(&params));		
	timer.Stop();

	// Debug
	GI_LOG("Total Used Nodes %d", 8 * (totalEightNodes - usedNodeCount));

	return timer.ElapsedMilliS();
}

double GISparseVoxelOctree::ConeTrace(GLuint depthBuffer,
									  GLuint normalBuffer,
									  GLuint colorBuffer,
									  const Camera& camera)
{
	return 0.0;
}

void GISparseVoxelOctree::LinkScene(GLuint lightBuffer,
									GLuint shadowMapArrayTexture)
{

}

uint64_t GISparseVoxelOctree::MemoryUsage() const
{
	uint64_t totalBytes = 0;
	totalBytes += dSVO.Size() * sizeof(CSVONode);
	totalBytes += dSVOMaterial.Size() * sizeof(CSVOMaterial);
	totalBytes += dSVOLock.Size() * sizeof(unsigned int);
	totalBytes += dSVOEmptyLoc.Size() * sizeof(unsigned int);
	totalBytes += sizeof(unsigned int);
	return totalBytes;
}