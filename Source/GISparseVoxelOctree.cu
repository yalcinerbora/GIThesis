#include "GISparseVoxelOctree.h"
#include <cuda_gl_interop.h>
#include "GICudaAllocator.h"
#include "GIKernels.cuh"
#include "CudaTimer.h"
#include "Macros.h"

GISparseVoxelOctree::GISparseVoxelOctree(GLuint lightIntensityTex)
	: dSVO()
	, lightIntensityTexLink(nullptr)
	, dSVONodeCountAtomic(1)
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
	for(unsigned int i = 0; i < allocatorSize; i++)
	{
		uint32_t depthMultiplier = 1;
		if(i == 0) depthMultiplier = (allocatorGrids[i].depth - GI_DENSE_LEVEL);
		totalAlloc += allocators[i]->NumPages() * GI_PAGE_SIZE * depthMultiplier;
	}
	dSVO.Resize(totalAlloc);
	dSVOColor.Resize(totalAlloc);

	dSVODense = dSVO.Data();
	dSVOSparse = dSVO.Data() + (GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);

	dSVOLevelStartIndices.Resize(allocatorGrids[0].depth + allocatorSize);
	dSVOLevelStartIndices.Memset(0x00, 0, dSVOLevelStartIndices.Size());

	unsigned int totalLevel = static_cast<unsigned int>(allocatorGrids[0].depth + allocatorSize - 1);

	hSVOConstants.denseDepth = GI_DENSE_LEVEL;
	hSVOConstants.denseDim = GI_DENSE_SIZE;
	hSVOConstants.totalDepth = totalLevel;
	hSVOConstants.numCascades = static_cast<unsigned int>(allocatorSize);

	// Copy to device
	CUDA_CHECK(cudaMemcpy(dSVOConstants.Data(), 
						  &hSVOConstants, 
						  sizeof(CSVOConstants), 
						  cudaMemcpyHostToDevice));


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

void GISparseVoxelOctree::ConstructDense()
{
	//double childSet, alloc;
	//CudaTimer timer;
	//timer.Start();

	// Level 0 is special it constructs the upper levels in addition to its level
	uint32_t gridSize = ((allocators[0]->NumPages() * GI_PAGE_SIZE) + 
						 GI_THREAD_PER_BLOCK - 1) /
						 GI_THREAD_PER_BLOCK;
	SVOReconstructChildSet<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVODense,
		allocators[0]->GetVoxelPagesDevice(),
		
		0u,
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	//timer.Stop();
	//childSet = timer.ElapsedMilliS();
	//timer.Start();

	gridSize = ((GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE) + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateNext<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVO.Data(),
		*dSVONodeCountAtomic.Data(),
		*dSVOLevelStartIndices.Data(),
		*dSVOLevelStartIndices.Data(),
		GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE
	);
	CUDA_KERNEL_CHECK();

	//timer.Stop();
	//alloc = timer.ElapsedMilliS();
	//timer.Start();

	// Copy Level Start Location to array
	CUDA_CHECK(cudaMemcpy(dSVOLevelStartIndices.Data() + 1,
						  dSVONodeCountAtomic.Data(),
						  sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	//GI_LOG("---------------------------------------");
	//GI_LOG("Level %d", GI_DENSE_LEVEL);
	//GI_LOG("Child %f ms", childSet);
	//GI_LOG("Alloc %f ms", alloc);
	//GI_LOG("");
}

void GISparseVoxelOctree::ConstructLevel(unsigned int currentLevel,
										 unsigned int allocatorIndex,
										 unsigned int cascadeNo)
{
	//double childSet, memCopy, alloc;
	//CudaTimer timer;
	//timer.Start();

	// Early Bail check 
	unsigned int currentLevelIndex = currentLevel - GI_DENSE_LEVEL;
	uint32_t levelNodeCount, levelNodeStarts[2];
	CUDA_CHECK(cudaMemcpy(levelNodeStarts,
						  dSVOLevelStartIndices.Data() + currentLevelIndex - 1,
						  sizeof(unsigned int) * 2,
						  cudaMemcpyDeviceToHost));
	levelNodeCount = levelNodeStarts[1] - levelNodeStarts[0];

	if(levelNodeCount == 0) return;

	// ChildBitSet your Level (with next level's child)
	// Allocate next level
	// Memcopy next level start location to array
	// Only ChildBitSet Upper Level
	// Then Allocate your level
	// Average Color to the level
	uint32_t gridSize = ((allocators[allocatorIndex]->NumPages() * GI_PAGE_SIZE) + 
						 GI_THREAD_PER_BLOCK - 1) /
						 GI_THREAD_PER_BLOCK;

	SVOReconstructChildSet<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVOSparse,
		tSVODense,
		allocators[allocatorIndex]->GetVoxelPagesDevice(),
		dSVOLevelStartIndices.Data(),

		cascadeNo,
		currentLevel,
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	//timer.Stop();
	//memCopy = timer.ElapsedMilliS();
	//timer.Start();

	gridSize = ((levelNodeCount) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateNext<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVOSparse,
		*dSVONodeCountAtomic.Data(),
		*(dSVOLevelStartIndices.Data() + currentLevelIndex - 1),
		*(dSVOLevelStartIndices.Data() + currentLevelIndex),
		levelNodeCount
	);
	CUDA_KERNEL_CHECK();

	//timer.Stop();
	//alloc = timer.ElapsedMilliS();
	//timer.Start();

	// Copy Level Start Location to array
	CUDA_CHECK(cudaMemcpy(dSVOLevelStartIndices.Data() + currentLevelIndex + 1, dSVONodeCountAtomic.Data(),
						  sizeof(unsigned int), cudaMemcpyDeviceToDevice));

//	dSVO.DumpToFile("svoDump", 0, levelNodeStarts[1] + GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
//	dSVOLevelStartIndices.DumpToFile("lvlDump");

	//GI_LOG("Level %d", currentLevel);
	//GI_LOG("Child %f ms", childSet);
	//GI_LOG("Alloc %f ms", alloc);
	//GI_LOG("Memcpy %f ms", memCopy);
	//GI_LOG("");
}

double GISparseVoxelOctree::UpdateSVO()
{
	CudaTimer timer;
	timer.Start();

	// Reset Atomic Counter since we reconstruct every frame
	unsigned int usedNodeCount;
	CUDA_CHECK(cudaMemcpy(&usedNodeCount, dSVONodeCountAtomic.Data(), sizeof(unsigned int),
						  cudaMemcpyDeviceToHost));
	dSVO.Memset(0x00, 0, usedNodeCount + GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
	dSVONodeCountAtomic.Memset(0x00, 0, 1);
	dSVOLevelStartIndices.Memset(0x00, 0, dSVOLevelStartIndices.Size());

	// Start with constructing dense
	ConstructDense();
	
	// Copy to dense
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
		
	//CudaVector<unsigned int> texRead;
	//texRead.Resize(GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
	//
	////dim3 blockSize(GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE);
	////fastreadkernel3d<<<1, blockSize>>>
	////(
	////	tSVODense,
	////	texRead.Data()
	////);

	//fastreadkernel<<<1, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE>>>
	//(
	//	tSVODense,
	//	texRead.Data()
	//);

//	texRead.DumpToFile("texDump");
//	dSVO.DumpToFile("svoDump", 0, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
//	dSVOLevelStartIndices.DumpToFile("lvlDump");

	// Construct Levels
	for(unsigned int i = GI_DENSE_LEVEL + 1; i < allocatorGrids[0].depth; i++)
	{
		ConstructLevel(i, 0, 0);
	}

	//// Now adding cascade levels
	//for(unsigned int i = 1; i < allocators.size(); i++)
	//{
	//	unsigned int currentLevel = allocatorGrids[0].depth + i;
	//	ConstructLevel(currentLevel, i, i);
	//}

	//DEBUG
	std::vector<unsigned int> nodeCounts;
	nodeCounts.resize(dSVOLevelStartIndices.Size());
	CUDA_CHECK(cudaMemcpy(nodeCounts.data(), dSVOLevelStartIndices.Data(),
		sizeof(unsigned int) * dSVOLevelStartIndices.Size(), cudaMemcpyDeviceToHost));

	GI_LOG("-------------------------------------------");
	GI_LOG("Tree Node Data");
	for(unsigned int i = 0; i <= allocatorGrids[0].depth - GI_DENSE_LEVEL; i++)
	{
		if(i == 0) GI_LOG("#%d Dense : %d", GI_DENSE_LEVEL + i, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
		else GI_LOG("#%d Level : %d", GI_DENSE_LEVEL + i, nodeCounts[i] - nodeCounts[i - 1]);
	}
	GI_LOG("-------------------------------------------");

	timer.Stop();
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

//VoxelDebugVAO GISparseVoxelOctree::VoxelDataForRendering(double& transferTime,
//														 unsigned int& voxelCount,
//														 unsigned int level)
//{
//	// Find Node count
//	unsigned int currentLevelIndex = level - GI_DENSE_LEVEL;
//	uint32_t levelNodeCount, levelNodeStarts[2];
//	CUDA_CHECK(cudaMemcpy(levelNodeStarts,
//						  dSVOLevelStartIndices.Data() + currentLevelIndex - 1,
//						  sizeof(unsigned int) * 2,
//						  cudaMemcpyDeviceToHost));
//	levelNodeCount = levelNodeStarts[1] - levelNodeStarts[0];
//
//	//
//	vaoNormPosData.Resize(levelNodeCount);
//	vaoColorData.Resize(levelNodeCount);
//
//	//// Cuda stuff;
//
//	////
//	//uint32_t gridSize = ((levelNodeCount) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
//	//SVOVoxelFetch<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//(
//	//	dSVOSparse,
//	//	*dSVONodeCountAtomic.Data(),
//	//	*(dSVOLevelStartIndices.Data() + currentLevelIndex - 1),
//	//	*(dSVOLevelStartIndices.Data() + currentLevelIndex),
//	//	levelNodeCount
//	//);
//	//CUDA_KERNEL_CHECK();
//	
//}

uint64_t GISparseVoxelOctree::MemoryUsage() const
{
	uint64_t totalBytes = 0;
	totalBytes += dSVO.Size() * sizeof(CSVONode);
	totalBytes += dSVOColor.Size() * sizeof(CSVOColor);
	totalBytes += dSVOLevelStartIndices.Size() * sizeof(unsigned int);
	totalBytes += sizeof(unsigned int);
	return totalBytes;
}