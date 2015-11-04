#include "GISparseVoxelOctree.h"
#include <cuda_gl_interop.h>
#include "GICudaAllocator.h"
#include "GIKernels.cuh"
#include "CudaTimer.h"
#include "Macros.h"

const unsigned int GISparseVoxelOctree::TPBWithHelperWarp = GI_THREAD_PER_BLOCK_PRIME + (32 - (GI_THREAD_PER_BLOCK_PRIME % 32));

GISparseVoxelOctree::GISparseVoxelOctree(GLuint lightIntensityTex)
	: dSVO()
	, lightIntensityTexLink(nullptr)
	, dSVONodeCountAtomic(1)
	, dSVOConstants(1)
	, tSVODense(0)
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
}

__global__ void fastreadkernel(cudaTextureObject_t texture,
							   unsigned int* gOut)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	uint3 texUV;
	texUV.x = globalId % GI_DENSE_SIZE;
	texUV.y = globalId / GI_DENSE_SIZE;
	texUV.z = globalId / (GI_DENSE_SIZE * GI_DENSE_SIZE);

	unsigned int currentNode = tex3D<unsigned int>(texture, texUV.x, texUV.y, texUV.z);
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

	// Texture of SVO Dense
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};

	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = dSVODense;
	resDesc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindUnsigned);
	resDesc.res.linear.sizeInBytes = sizeof(unsigned int) * GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE;

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
						 TPBWithHelperWarp - 1) /
						 TPBWithHelperWarp;
	SVOReconstructChildSet<<<gridSize, TPBWithHelperWarp>>>
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

	// ChildBitSet your Level
	// Allocate next level
	// Memcopy next level start location to array
	// Only ChildBitSet Upper Level
	// Then Allocate your level
	// Average Color to the level
	unsigned int currentLevelIndex = currentLevel - GI_DENSE_LEVEL;
	uint32_t gridSize = ((allocators[allocatorIndex]->NumPages() * GI_PAGE_SIZE) + 
						 TPBWithHelperWarp - 1) /
						 TPBWithHelperWarp;

	SVOReconstructChildSet<<<gridSize, TPBWithHelperWarp>>>
	(
		dSVOSparse,
		tSVODense,
		//dSVODense,
		allocators[allocatorIndex]->GetVoxelPagesDevice(),
		dSVOLevelStartIndices.Data(),

		cascadeNo,
		currentLevel,
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	//timer.Stop();
	//childSet = timer.ElapsedMilliS();
	//timer.Start();

	// Call count is on GPU
	uint32_t levelNodeCount, levelNodeStarts[2];
	CUDA_CHECK(cudaMemcpy(levelNodeStarts, 
						  dSVOLevelStartIndices.Data() + currentLevelIndex - 1, 
						  sizeof(unsigned int) * 2,
						  cudaMemcpyDeviceToHost));
	levelNodeCount = levelNodeStarts[1] - levelNodeStarts[0];

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
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

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
	
	cudaDeviceSynchronize();

	// Texture of SVO Dense
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};

	resDesc.resType = cudaResourceTypeLinear;
	resDesc.res.linear.devPtr = dSVODense;
	resDesc.res.linear.desc = cudaCreateChannelDesc<unsigned int>();
	resDesc.res.linear.sizeInBytes = sizeof(unsigned int) * GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE;

	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	if(tSVODense != 0) CUDA_CHECK(cudaDestroyTextureObject(tSVODense));
	CUDA_CHECK(cudaCreateTextureObject(&tSVODense, &resDesc, &texDesc, nullptr));


	CudaVector<unsigned int> texRead;
	texRead.Resize(GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
	fastreadkernel<<<GI_DENSE_SIZE * GI_DENSE_SIZE, GI_DENSE_SIZE>>>
	(
		tSVODense,
		texRead.Data()
	);

	texRead.DumpToFile("texDump");
	dSVO.DumpToFile("svoDump", 0, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
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
	//std::vector<unsigned int> nodeCounts;
	//nodeCounts.resize(dSVOLevelStartIndices.Size());
	//CUDA_CHECK(cudaMemcpy(nodeCounts.data(), dSVOLevelStartIndices.Data(),
	//	sizeof(unsigned int) * dSVOLevelStartIndices.Size(), cudaMemcpyDeviceToHost));

	//GI_LOG("-------------------------------------------");
	//GI_LOG("Tree Node Data");
	//for(unsigned int i = 0; i <= allocatorGrids[0].depth - GI_DENSE_LEVEL; i++)
	//{
	//	if(i == 0) GI_LOG("#%d Dense : %d", GI_DENSE_LEVEL + i, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
	//	else GI_LOG("#%d Level : %d", GI_DENSE_LEVEL + i, nodeCounts[i] - nodeCounts[i - 1]);
	//}
	//GI_LOG("-------------------------------------------");

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

uint64_t GISparseVoxelOctree::MemoryUsage() const
{
	uint64_t totalBytes = 0;
	totalBytes += dSVO.Size() * sizeof(CSVONode);
	totalBytes += dSVOColor.Size() * sizeof(CSVOColor);
	totalBytes += dSVOLevelStartIndices.Size() * sizeof(unsigned int);
	totalBytes += sizeof(unsigned int);
	return totalBytes;
}