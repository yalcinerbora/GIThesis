#include "GISparseVoxelOctree.h"
#include <cuda_gl_interop.h>
#include "GICudaAllocator.h"
#include "GIKernels.cuh"
#include "CudaTimer.h"

const unsigned int GISparseVoxelOctree::TPBWithHelperWarp = GI_THREAD_PER_BLOCK_PRIME + (32 - (GI_THREAD_PER_BLOCK_PRIME % 32));

GISparseVoxelOctree::GISparseVoxelOctree(GLuint lightIntensityTex)
	: dSVO()
	, lightIntensityTexLink(nullptr)
	, dSVONodeCountAtomic(1)
	, dSVOConstants(1)
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

	dSVOLevelStartIndices.Resize(allocatorGrids[0].depth + allocatorSize - 1);
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
}

void GISparseVoxelOctree::ConstructDense()
{
	// Level 0 is special it constructs the upper levels in addition to its level
	uint32_t gridSize = ((allocators[0]->NumPages() * GI_PAGE_SIZE) + TPBWithHelperWarp - 1) /
						 TPBWithHelperWarp;
	SVOReconstructChildSet<<<gridSize, TPBWithHelperWarp>>>
	(
		dSVODense,
		allocators[0]->GetVoxelPagesDevice(),
		
		0u,
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	dSVO.DumpToFile("svoDump", 0, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);

	gridSize = ((GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE) + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateNext<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVO.Data(),
		*dSVONodeCountAtomic.Data(),
		*dSVOLevelStartIndices.Data(),
		GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE
	);
	CUDA_KERNEL_CHECK();

	// Copy Level Start Location to array
	CUDA_CHECK(cudaMemcpy(dSVOLevelStartIndices.Data() + 1,
						  dSVONodeCountAtomic.Data(),
						  sizeof(unsigned int), cudaMemcpyDeviceToDevice));

	dSVO.DumpToFile("svoDump", 0, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
}

void GISparseVoxelOctree::ConstructLevel(unsigned int currentLevel,
										 unsigned int allocatorIndex,
										 unsigned int cascadeNo)
{
	// ChildBitSet your Level
	// Allocate next level
	// Memcopy next level start location to array
	// Only ChildBitSet Upper Level
	// Then Allocate your level
	// Average Color to the level
	unsigned int currentLevelIndex = currentLevel - GI_DENSE_LEVEL;
	uint32_t gridSize = ((allocators[allocatorIndex]->NumPages() * GI_PAGE_SIZE) + TPBWithHelperWarp - 1) /
						 TPBWithHelperWarp;

	SVOReconstructChildSet<<<gridSize, TPBWithHelperWarp>>>
	(
		dSVOSparse,
		dSVODense,
		allocators[allocatorIndex]->GetVoxelPagesDevice(),
		dSVOLevelStartIndices.Data(),

		cascadeNo,
		currentLevel,
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	// Call count is on GPU
	uint32_t levelNodeCount, levelNodeStarts[2];
	CUDA_CHECK(cudaMemcpy(levelNodeStarts, 
						  dSVOLevelStartIndices.Data() + currentLevelIndex - 1, 
						  sizeof(unsigned int) * 2,
						  cudaMemcpyDeviceToHost));
	levelNodeCount = levelNodeStarts[1] - levelNodeStarts[0];


	dSVO.DumpToFile("svoDump", 0, levelNodeStarts[1] +
					GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);


	gridSize = ((levelNodeCount) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateNext<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVOSparse,
		*dSVONodeCountAtomic.Data(),
		*(dSVOLevelStartIndices.Data() + currentLevelIndex - 1),
		levelNodeCount
	);
	CUDA_KERNEL_CHECK();

	dSVO.DumpToFile("svoDump", 0, levelNodeStarts[1] + 
					GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);

	// Copy Level Start Location to array
	CUDA_CHECK(cudaMemcpy(dSVOLevelStartIndices.Data() + currentLevelIndex + 1, dSVONodeCountAtomic.Data(),
						  sizeof(unsigned int), cudaMemcpyDeviceToDevice));
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
	
	//DEBUG
	dSVO.DumpToFile("svoDump", 0, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
	dSVOLevelStartIndices.DumpToFile("startIndices");

	// Construct Levels
	for(unsigned int i = GI_DENSE_LEVEL + 1; i < allocatorGrids[0].depth; i++)
	{
		ConstructLevel(i, 0, 0);

		//DEBUG
		dSVOLevelStartIndices.DumpToFile("startIndices");
	}

	//// Now adding cascade levels
	//for(unsigned int i = 1; i < allocators.size(); i++)
	//{
	//	unsigned int currentLevel = allocatorGrids[0].depth + i;
	//	ConstructLevel(currentLevel, i, i);
	//}

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

//void GICudaAllocator::LinkSceneShadowMapArray(GLuint shadowMapArray)
//{
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&sceneShadowMapLink,
//	//									   shadowMapArray,
//	//									   GL_TEXTURE_2D_ARRAY,
//	//									   cudaGraphicsRegisterFlagsReadOnly));
//}
//
//void GICudaAllocator::LinkSceneGBuffers(GLuint depthTex,
//										GLuint normalTex,
//										GLuint lightIntensityTex)
//{
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&depthBuffLink,
//	//										depthTex,
//	//										GL_TEXTURE_2D,
//	//										cudaGraphicsRegisterFlagsReadOnly));
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&normalBuffLink,
//	//										normalTex,
//	//										GL_TEXTURE_2D,
//	//										cudaGraphicsRegisterFlagsReadOnly));
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&lightIntensityLink,
//	//										lightIntensityTex,
//	//										GL_TEXTURE_2D,
//	//										cudaGraphicsRegisterFlagsSurfaceLoadStore));
//}
//
//void GICudaAllocator::UnLinkGBuffers()
//{
//	//CUDA_CHECK(cudaGraphicsUnregisterResource(depthBuffLink));
//	//CUDA_CHECK(cudaGraphicsUnregisterResource(normalBuffLink));
//	//CUDA_CHECK(cudaGraphicsUnregisterResource(lightIntensityLink));
//}

// Textures
//cudaArray_t texArray;
//cudaMipmappedArray_t mipArray;
//cudaResourceDesc resDesc = {};
//cudaTextureDesc texDesc = {};

//resDesc.resType = cudaResourceTypeMipmappedArray;

//texDesc.addressMode[0] = cudaAddressModeWrap;
//texDesc.addressMode[1] = cudaAddressModeWrap;
//texDesc.filterMode = cudaFilterModePoint;
//texDesc.readMode = cudaReadModeElementType;
//texDesc.normalizedCoords = 1;

//CUDA_CHECK(cudaGraphicsMapResources(1, &sceneShadowMapLink));
//CUDA_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&mipArray, sceneShadowMapLink));
//resDesc.res.mipmap.mipmap = mipArray;
//CUDA_CHECK(cudaCreateTextureObject(&shadowMaps, &resDesc, &texDesc, nullptr));

//texDesc.normalizedCoords = 1;
//resDesc.resType = cudaResourceTypeArray;

//CUDA_CHECK(cudaGraphicsMapResources(1, &depthBuffLink));
//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, depthBuffLink, 0, 0));
//resDesc.res.array.array = texArray;
//CUDA_CHECK(cudaCreateTextureObject(&depthBuffer, &resDesc, &texDesc, nullptr));

//CUDA_CHECK(cudaGraphicsMapResources(1, &normalBuffLink));
//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, normalBuffLink, 0, 0));
//resDesc.res.array.array = texArray;
//CUDA_CHECK(cudaCreateTextureObject(&normalBuffer, &resDesc, &texDesc, nullptr));

//CUDA_CHECK(cudaGraphicsMapResources(1, &lightIntensityLink));
//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&texArray, lightIntensityLink, 0, 0));
//resDesc.res.array.array = texArray;
//CUDA_CHECK(cudaCreateSurfaceObject(&lightIntensityBuffer, &resDesc));

