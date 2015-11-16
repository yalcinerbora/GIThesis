#include "GISparseVoxelOctree.h"
#include <cuda_gl_interop.h>
#include "GICudaAllocator.h"
#include "GIKernels.cuh"
#include "CudaTimer.h"
#include "Macros.h"

GISparseVoxelOctree::GISparseVoxelOctree()
	: svoNodeBuffer(512)
	, svoMaterialBuffer(512)
	, dSVONodeAllocator(1)
	, dSVOConstants(1)
	, tSVODense(0)
	, vaoNormPosData(512)
	, vaoColorData(512)
	, computeVoxTraceWorld(ShaderType::COMPUTE, "Shaders/VoxTraceWorld.glsl")
{}

GISparseVoxelOctree::~GISparseVoxelOctree()
{
	if(svoNodeResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoNodeResource));
	if(svoMaterialResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoMaterialResource));
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

	unsigned int currentNode = tex3D<unsigned int>(texture, texUV.x, texUV.y, texUV.z);
	printf("%d TexCoordXYZ %f, %f, %f\n", globalId, texUV.x, texUV.y, texUV.z);

	gOut[globalId] = currentNode;
}

void GISparseVoxelOctree::LinkAllocators(GICudaAllocator** newAllocators,
										 size_t allocatorSize,
										 float sceneMultiplier)
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
	size_t totalAlloc = static_cast<size_t>(sceneMultiplier * 1024.0f * 1024.0f);
	svoNodeBuffer.Resize(totalAlloc + GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);

	dSVODense = nullptr;
	dSVOSparse = nullptr;

	// Mat Tree holds up to level 0
	matSparseOffset = static_cast<unsigned int>((1.0 - std::pow(8.0f, GI_DENSE_LEVEL + 1)) / (1.0f - 8.0f));
	svoMaterialBuffer.Resize(totalAlloc + matSparseOffset);

	hSVOLevelOffsets.resize(allocatorGrids[0].depth + allocatorSize - GI_DENSE_LEVEL + 2);
	hSVOLevelSizes.resize(allocatorGrids[0].depth + allocatorSize - GI_DENSE_LEVEL);
	dSVOLevelSizes.Resize(allocatorGrids[0].depth + allocatorSize - GI_DENSE_LEVEL);
	
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoNodeResource, 
											svoNodeBuffer.getGLBuffer(), 
											cudaGLMapFlagsWriteDiscard));

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoMaterialResource, 
											svoMaterialBuffer.getGLBuffer(), 
											cudaGLMapFlagsWriteDiscard));

	// Clear All Data
	GLuint allOne = 0xFFFFFFFF;
	GLuint zero = 0;

	glBindBuffer(GL_COPY_WRITE_BUFFER, svoNodeBuffer.getGLBuffer());
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &allOne);

	glBindBuffer(GL_COPY_WRITE_BUFFER, svoMaterialBuffer.getGLBuffer());
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

	dSVONodeAllocator.Memset(0x00, 0, 1);
	dSVOLevelSizes.Memset(0x00, 0, dSVOLevelSizes.Size());
	std::fill(hSVOLevelSizes.begin(), hSVOLevelSizes.end(), 0);
	std::fill(hSVOLevelOffsets.begin(), hSVOLevelOffsets.end(), 0);
	
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
	// Level 0 does not gurantee lower cascade parents
	// Each Allocator tries to allocate its parent
	for(unsigned int i = 0; i < allocators.size(); i++)
	{
		uint32_t gridSize = ((allocators[i]->NumPages() * GI_PAGE_SIZE) + 
							GI_THREAD_PER_BLOCK - 1) /
							GI_THREAD_PER_BLOCK;
		SVOReconstructDetermineNode<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(
			dSVODense,
			allocators[i]->GetVoxelPagesDevice(),
		
			i,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}

	uint32_t gridSize = ((GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE) + GI_THREAD_PER_BLOCK - 1) /
						  GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateLevel<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVODense,
		dSVOLevelSizes.Data(),
		*dSVONodeAllocator.Data(),

		0,
		static_cast<unsigned int>(svoNodeBuffer.Capacity() - GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE),
		GI_DENSE_LEVEL,
		GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE,
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	// Copy Level Start Location to array
	CUDA_CHECK(cudaMemcpy(hSVOLevelOffsets.data() + 2,
							dSVONodeAllocator.Data(),
							sizeof(unsigned int), 
							cudaMemcpyDeviceToHost));
}

void GISparseVoxelOctree::ConstructLevel(unsigned int currentLevel,
										 unsigned int allocatorOffset)
{
	// Early Bail check 
	unsigned int currentLevelIndex = currentLevel - GI_DENSE_LEVEL;
	CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data() + currentLevelIndex,
						  dSVOLevelSizes.Data() + currentLevelIndex,
						  sizeof(unsigned int),
						  cudaMemcpyDeviceToHost));
	if(hSVOLevelSizes[currentLevelIndex] == 0) return;

	// ChildBitSet your Level (with next level's child)
	// Allocate next level
	// Memcopy next level start location to array
	// Only ChildBitSet Upper Level
	// Then Allocate your level
	// Average Color to the level
	for(unsigned int i = allocatorOffset; i < allocators.size(); i++)
	{
		uint32_t gridSize = ((allocators[i]->NumPages() * GI_PAGE_SIZE) + 
							 GI_THREAD_PER_BLOCK - 1) /
							 GI_THREAD_PER_BLOCK;

		SVOReconstructDetermineNode<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(
			dSVOSparse,
			tSVODense,
			allocators[i]->GetVoxelPagesDevice(),

			i,
			currentLevel,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}
	
	uint32_t gridSize = (hSVOLevelSizes[currentLevelIndex] + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateLevel<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVOSparse,
		dSVOLevelSizes.Data(),
		*dSVONodeAllocator.Data(),

		hSVOLevelOffsets[currentLevelIndex],
		static_cast<unsigned int>(svoNodeBuffer.Capacity() - GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE),
		currentLevel,
		hSVOLevelSizes[currentLevelIndex],
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();

	

	// Copy Level Start Location to array
	CUDA_CHECK(cudaMemcpy(hSVOLevelOffsets.data() + ((currentLevel - GI_DENSE_LEVEL) + 2),
						  dSVONodeAllocator.Data(),
						  sizeof(unsigned int),
						  cudaMemcpyDeviceToHost));
}

void GISparseVoxelOctree::ConstructFullAtomic()
{
	// Fully Atomic Version
	for(unsigned int i = 0; i < allocators.size(); i++)
	{
		uint32_t gridSize = (allocators[i]->NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
							GI_THREAD_PER_BLOCK;
		SVOReconstruct<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(
			dSVOMaterial,
			dSVOSparse,
			dSVODense,
			dSVOLevelSizes.Data(),
			*dSVONodeAllocator.Data(),

			allocators[i]->GetVoxelPagesDevice(),
			allocators[i]->GetObjRenderCacheDevice(),

			matSparseOffset,
			static_cast<unsigned int>(svoNodeBuffer.Capacity() - (GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE)),
			i,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}

	// Reduce Colors to lower levels
	// TODO:
	


	
	// Copy Dense to Texture
	cudaMemcpy3DParms params = { 0 };
	params.dstArray = denseArray;
	params.srcPtr =
	{
		dSVODense,
		GI_DENSE_SIZE * sizeof(unsigned int),
		GI_DENSE_SIZE,
		GI_DENSE_SIZE
	};
	params.extent = { GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE };
	params.kind = cudaMemcpyDeviceToDevice;
	CUDA_CHECK(cudaMemcpy3D(&params));
}

void GISparseVoxelOctree::ConstructLevelByLevel()
{
	// Start with constructing dense
	ConstructDense();

	// Copy Dense to Texture
	cudaMemcpy3DParms params = { 0 };
	params.dstArray = denseArray;
	params.srcPtr =
	{
		dSVODense,
		GI_DENSE_SIZE * sizeof(unsigned int),
		GI_DENSE_SIZE,
		GI_DENSE_SIZE
	};
	params.extent = { GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE };
	params.kind = cudaMemcpyDeviceToDevice;
	CUDA_CHECK(cudaMemcpy3D(&params));

	// Construct Levels
	for(unsigned int i = GI_DENSE_LEVEL + 1; i < allocatorGrids[0].depth; i++)
	{
		ConstructLevel(i, 0);
	}

	// Now adding cascade levels
	for(unsigned int i = 1; i < allocators.size(); i++)
	{
		unsigned int currentLevel = allocatorGrids[0].depth + i - 1;
		ConstructLevel(currentLevel, i);
	}

	// Last memcpy of the leaf cascade size
	CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data() + (allocatorGrids[0].depth - GI_DENSE_LEVEL),
		dSVOLevelSizes.Data() + (allocatorGrids[0].depth - GI_DENSE_LEVEL),
		sizeof(unsigned int),
		cudaMemcpyDeviceToHost));
}

void GISparseVoxelOctree::AverageNodes(bool orderedNodes)
{
	// Leaf Nodes Already ordered
	
	// First Average Leafs atomically	
	//for(unsigned int i = 0; i < allocators.size(); i++)
	//{
	//	assert(allocators[i]->IsGLMapped() == true);
	//	uint32_t gridSize = (allocators[i]->NumPages() * GI_PAGE_SIZE +  GI_THREAD_PER_BLOCK - 1) / 
	//						GI_THREAD_PER_BLOCK;
	//			
	//	// Average Leaf Node
	//	SVOReconstructAverageLeaf<<<gridSize, GI_THREAD_PER_BLOCK>>>
	//	(
	//		dSVOMaterial.Data(),
	//		dSVOSparse,
	//		tSVODense,
	//		allocators[i]->GetVoxelPagesDevice(),
	//		dSVOLevelStartIndices.Data(),
	//		allocators[i]->GetObjRenderCacheDevice(),
	//		matSparseOffset,
	//		i,
	//		hSVOConstants.totalDepth - (hSVOConstants.numCascades - i),
	//		*dSVOConstants.Data()
	//	);
	//	CUDA_KERNEL_CHECK();
	//}

	// Now use leaf nodes to average upper nodes
	// Start bottom up (dont average until inner averages itself
	// TODO
}

double GISparseVoxelOctree::UpdateSVO()
{
	CUDA_CHECK(cudaGraphicsMapResources(1, &svoMaterialResource));
	CUDA_CHECK(cudaGraphicsMapResources(1, &svoNodeResource));
	
	size_t size;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVODense), 
													 &size, svoNodeResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVOMaterial),
													 &size, svoMaterialResource));
	dSVOSparse = dSVODense + GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE;

	CudaTimer timer;
	timer.Start();

	// Reset Atomic Counter since we reconstruct every frame
	unsigned int usedNodeCount;
	CUDA_CHECK(cudaMemcpy(&usedNodeCount, dSVONodeAllocator.Data(), sizeof(unsigned int),
						  cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemset(dSVODense, 0xFF, sizeof(CSVONode) * (usedNodeCount + GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE)));
	CUDA_CHECK(cudaMemset(dSVOMaterial + matSparseOffset, 0x00, sizeof(CSVOMaterial) * usedNodeCount));

	dSVONodeAllocator.Memset(0x00, 0, 1);
	dSVOLevelSizes.Memset(0x00, 0, dSVOLevelSizes.Size());
	std::fill(hSVOLevelSizes.begin(), hSVOLevelSizes.end(), 0);
	std::fill(hSVOLevelOffsets.begin(), hSVOLevelOffsets.end(), 0);

	// Maxwell is faster with fully atomic code (CAS Locks etc.)
	// However kepler sucks (100ms compared to 5ms) 
	if(CudaInit::CapabilityMajor() >= 5)
	{
		// Since fully atomic construction does not 
		// create level nodes in ordered manner
		// for each level we need to traverse node
		ConstructFullAtomic();
		AverageNodes(false);
	}
	else
	{
		ConstructLevelByLevel();
		AverageNodes(true);
	}

	//// DEBUG
	//GI_LOG("-------------------------------------------");
	//GI_LOG("Tree Node Data");
	//CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data(),
	//					  dSVOLevelSizes.Data(),
	//					  dSVOLevelSizes.Size() * sizeof(unsigned int),
	//					  cudaMemcpyDeviceToHost));
	//unsigned int i;
	//for(i = 0; i <= allocatorGrids[0].depth - GI_DENSE_LEVEL + allocators.size() - 1; i++)
	//{
	//	if(i == 0) GI_LOG("#%d Dense : %d", GI_DENSE_LEVEL + i, GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE);
	//	else GI_LOG("#%d Level : %d", GI_DENSE_LEVEL + i, hSVOLevelSizes[i]);
	//}
	//unsigned int total;
	//CUDA_CHECK(cudaMemcpy(&total, dSVONodeAllocator.Data(), sizeof(unsigned int),
	//					  cudaMemcpyDeviceToHost));
	//GI_LOG("Total : %d", total);
	//GI_LOG("-------------------------------------------");

	timer.Stop();
	
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoMaterialResource));
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoNodeResource));
	return timer.ElapsedMilliS();
}

double GISparseVoxelOctree::ConeTrace(GLuint depthBuffer,
									  GLuint normalBuffer,
									  GLuint colorBuffer,
									  const Camera& camera)
{
	return 0.0;
}

double GISparseVoxelOctree::SVODataToGL(// GL buffer ptrs
										CVoxelNormPos* dVAONormPosData,
										uint32_t* dVAOColorData,

										CVoxelGrid& voxGridData,
										uint32_t& voxCount,
										uint32_t level,
										uint32_t maxVoxelCount)
{
	// This version is for fully atomic level fetch
	// used for debugging

	// Since we do not know where specific level starts and ends on the voxel system
	// we ill launch entire node system




	return 0.0;
}

uint64_t GISparseVoxelOctree::MemoryUsage() const
{
	uint64_t totalBytes = 0;
	totalBytes += svoNodeBuffer.Capacity() * sizeof(CSVONode);
	totalBytes += svoMaterialBuffer.Capacity() * sizeof(CSVOMaterial);
	totalBytes += dSVOLevelSizes.Size() * sizeof(unsigned int);
	totalBytes += sizeof(unsigned int);
	return totalBytes;
}