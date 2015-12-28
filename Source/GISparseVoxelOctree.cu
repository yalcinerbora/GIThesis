#include "GISparseVoxelOctree.h"
#include "GICudaAllocator.h"
#include "GIKernels.cuh"
#include "CudaTimer.h"
#include "Macros.h"
#include "Camera.h"
#include "Globals.h"
#include "IEUtility/IEMath.h"
#include "DeferredRenderer.h"

#include <cuda_gl_interop.h>
#include <numeric>
#include <cuda_profiler_api.h>

const GLsizei GISparseVoxelOctree::TraceWidth = /*160;*//*320;*//*640;*//*800;*/1280;/*1920;*//*2560;*///3840;
const GLsizei GISparseVoxelOctree::TraceHeight = /*90;*//*180;*//*360;*//*450;*/720;/*1080;*//*1440;*///2160;

GISparseVoxelOctree::GISparseVoxelOctree()
	: svoNodeBuffer(512)
	, svoMaterialBuffer(512)
	, svoLevelOffsets(32)
	, dSVOConstants(1)
	, tSVODense(0)
	, computeVoxTraceWorld(ShaderType::COMPUTE, "Shaders/VoxTraceWorld.glsl")
	, computeVoxTraceDeferred(ShaderType::COMPUTE, "Shaders/VoxTraceDeferred.glsl")
	, computeAO(ShaderType::COMPUTE, "Shaders/VoxTraceAO.glsl")
	, computeAOSurf(ShaderType::COMPUTE, "Shaders/VoxSurfAO.glsl")
	, svoTraceData(1)
	, svoConeParams(1)
	, svoNodeResource(nullptr)
	, svoLevelOffsetResource(nullptr)
	, svoMaterialResource(nullptr)
	, liTexture(0)
{
	svoTraceData.AddData({});
	svoConeParams.AddData({});

	// Light Intensity Tex
	glGenTextures(1, &liTexture);
	glBindTexture(GL_TEXTURE_2D, liTexture);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, TraceWidth, TraceHeight);
}

GISparseVoxelOctree::~GISparseVoxelOctree()
{
	if(svoNodeResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoNodeResource));
	if(svoMaterialResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoMaterialResource));
	if(svoLevelOffsetResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoLevelOffsetResource));
	if(tSVODense) CUDA_CHECK(cudaDestroyTextureObject(tSVODense));
	if(denseArray) CUDA_CHECK(cudaFreeArray(denseArray));
	if(liTexture) glDeleteTextures(1, &liTexture);
}

void GISparseVoxelOctree::LinkAllocators(Array32<GICudaAllocator*> newAllocators,
										 uint32_t totalCount,
										 const uint32_t levelCounts[])
{
	allocatorGrids.clear();
	allocators.resize(newAllocators.length);
	allocatorGrids.resize(newAllocators.length);

	assert(newAllocators.length > 0);
	assert(newAllocators.arr != nullptr);

	std::copy(newAllocators.arr, newAllocators.arr + newAllocators.length, allocators.data());
	for(unsigned int i = 0; i < newAllocators.length; i++)
		allocatorGrids[i] = &(newAllocators.arr[i]->GetVoxelGridHost());

	size_t sparseNodeCount = allocatorGrids[0]->depth + newAllocators.length - GI_DENSE_LEVEL;
	uint32_t totalLevel = allocatorGrids[0]->depth + newAllocators.length - 1;
	size_t totalAlloc = totalCount;

	// TODO: More Dynamic Allocation Scheme
	hSVOLevelTotalSizes.resize(sparseNodeCount);
	dSVOLevelTotalSizes.Resize(sparseNodeCount);
	dSVOLevelSizes.Resize(sparseNodeCount);
	hSVOLevelSizes.resize(sparseNodeCount);
	
	svoNodeBuffer.Resize(totalAlloc + GI_DENSE_SIZE_CUBE);
	svoLevelOffsets.Resize(sparseNodeCount);

	dSVODense = nullptr;
	dSVOSparse = nullptr;

	// Mat Tree holds up to level 0
	matSparseOffset = static_cast<unsigned int>((1.0 - std::pow(8.0f, GI_DENSE_LEVEL + 1)) / 
												(1.0f - 8.0f));
	svoMaterialBuffer.Resize(totalAlloc + matSparseOffset);
	
	// Register
	if(svoNodeResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoNodeResource));
	if(svoMaterialResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoMaterialResource));
	if(svoLevelOffsetResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoLevelOffsetResource));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoNodeResource, 
											svoNodeBuffer.getGLBuffer(), 
											cudaGLMapFlagsWriteDiscard));

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoMaterialResource, 
											svoMaterialBuffer.getGLBuffer(), 
											cudaGLMapFlagsWriteDiscard));

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoLevelOffsetResource,
											svoLevelOffsets.getGLBuffer(),
											cudaGLMapFlagsReadOnly));

	// Actual Data Init
	GLuint allOne = 0xFFFFFFFF;
	GLuint zero = 0;

	glBindBuffer(GL_COPY_WRITE_BUFFER, svoNodeBuffer.getGLBuffer());
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &allOne);

	glBindBuffer(GL_COPY_WRITE_BUFFER, svoMaterialBuffer.getGLBuffer());
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);

	
	dSVOLevelSizes.Memset(0x00, 0, dSVOLevelSizes.Size());
	std::fill(hSVOLevelSizes.begin(), hSVOLevelSizes.end(), 0);
	std::copy(levelCounts + GI_DENSE_LEVEL, 
			  levelCounts + GI_DENSE_LEVEL + sparseNodeCount, 
			  hSVOLevelTotalSizes.data());
	hSVOLevelTotalSizes[0] = GI_DENSE_SIZE_CUBE;
	dSVOLevelTotalSizes = hSVOLevelTotalSizes;

	// SVO Constants set
	hSVOConstants.denseDepth = GI_DENSE_LEVEL;
	hSVOConstants.denseDim = GI_DENSE_SIZE;
	hSVOConstants.totalDepth = totalLevel;
	hSVOConstants.numCascades = newAllocators.length;

	// Offset Set
	uint32_t levelOffset = 0;
	svoLevelOffsets.CPUData().clear();
	for(unsigned int i = GI_DENSE_LEVEL; i <= totalLevel; i++)
	{
		svoLevelOffsets.AddData(levelOffset);
		levelOffset += (i != GI_DENSE_LEVEL) ? levelCounts[i] : 0;
	}
	svoLevelOffsets.SendData();
	assert(levelOffset <= totalCount);

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

	uint32_t gridSize = ((GI_DENSE_SIZE_CUBE) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateLevel<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVODense,
		*(dSVOLevelSizes.Data() + 1),
		*(dSVOLevelTotalSizes.Data() + 1),
		*(dSVOLevelTotalSizes.Data()),
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();
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
			dSVOOffsets,

			i,
			currentLevel,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}
	
	uint32_t gridSize = (hSVOLevelSizes[currentLevelIndex] + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	SVOReconstructAllocateLevel<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		dSVOSparse + svoLevelOffsets.CPUData()[currentLevelIndex],
		*(dSVOLevelSizes.Data() + currentLevelIndex + 1),
		*(dSVOLevelTotalSizes.Data() + currentLevelIndex + 1),
		*(dSVOLevelSizes.Data() + currentLevelIndex),
		*dSVOConstants.Data()
	);
	CUDA_KERNEL_CHECK();
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

			dSVOOffsets,
			dSVOLevelTotalSizes.Data(),
				
			// VoxelSystem Data
			allocators[i]->GetVoxelPagesDevice(),
			allocators[i]->GetObjRenderCacheDevice(),

			matSparseOffset,
			i,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}
	// Copy Level Sizes
	CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data(),
						  dSVOLevelSizes.Data(),
						  hSVOLevelSizes.size() * sizeof(uint32_t),
						  cudaMemcpyDeviceToHost));
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
	for(unsigned int i = GI_DENSE_LEVEL + 1; i < allocatorGrids[0]->depth; i++)
	{
		ConstructLevel(i, 0);
	}

	// Now adding cascade levels
	for(unsigned int i = 1; i < allocators.size(); i++)
	{
		unsigned int currentLevel = allocatorGrids[0]->depth + i - 1;
		ConstructLevel(currentLevel, i);
	}

	// Memcpy Last Total Size
	CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data() + (hSVOConstants.totalDepth - GI_DENSE_LEVEL),
						  dSVOLevelSizes.Data() + (hSVOConstants.totalDepth - GI_DENSE_LEVEL),
						  sizeof(uint32_t),
						  cudaMemcpyDeviceToHost));
}

void GISparseVoxelOctree::AverageNodes(bool skipLeaf)
{
	// First Average Leafs atomically
	if(!skipLeaf)
	for(unsigned int i = 0; i < allocators.size(); i++)
	{
		assert(allocators[i]->IsGLMapped() == true);
		uint32_t gridSize = (allocators[i]->NumPages() * GI_PAGE_SIZE +  GI_THREAD_PER_BLOCK - 1) / 
							GI_THREAD_PER_BLOCK;
				
		// Average Leaf Node
		SVOReconstructMaterialLeaf<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(
			dSVOMaterial,

			// Const SVO Data
			dSVOSparse,
			dSVOOffsets,
			tSVODense,

			// Page Data
			allocators[i]->GetVoxelPagesDevice(),
										  
			// For Color Lookup
			allocators[i]->GetObjRenderCacheDevice(),

			// Constants
			matSparseOffset,
			i,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}

	// Now use leaf nodes to average upper nodes
	// Start bottom up
	for(int i = hSVOConstants.totalDepth - 1; i >= static_cast<int>(hSVOConstants.denseDepth); i--)
	{
		unsigned int arrayIndex = i - GI_DENSE_LEVEL;
		unsigned int levelDim = GI_DENSE_SIZE >> (GI_DENSE_LEVEL - i);
		unsigned int levelSize = (i > GI_DENSE_LEVEL) ? hSVOLevelSizes[arrayIndex] : 
														levelDim * levelDim * levelDim;
		if(levelSize == 0) continue;

		uint32_t gridSize = ((levelSize * GI_NODE_THREAD_COUNT) + GI_THREAD_PER_BLOCK - 1) /
							GI_THREAD_PER_BLOCK;
		
		// Average Level
		SVOReconstructAverageNode<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(
			dSVOMaterial,
			dSVODense,
			dSVOSparse,

			*(dSVOOffsets + arrayIndex),
			*(dSVOOffsets + arrayIndex + 1),

			levelSize,
			matSparseOffset,
			i,
			*dSVOConstants.Data()
		);
		CUDA_KERNEL_CHECK();
	}
	// Call once for all lower levels
}

double GISparseVoxelOctree::UpdateSVO()
{
	CUDA_CHECK(cudaProfilerStart());

	CUDA_CHECK(cudaGraphicsMapResources(1, &svoMaterialResource));
	CUDA_CHECK(cudaGraphicsMapResources(1, &svoNodeResource));
	CUDA_CHECK(cudaGraphicsMapResources(1, &svoLevelOffsetResource));
	
	size_t size;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVODense), 
													 &size, svoNodeResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVOMaterial),
													 &size, svoMaterialResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVOOffsets),
													&size, svoLevelOffsetResource));
	dSVOSparse = dSVODense + GI_DENSE_SIZE_CUBE;

	CudaTimer timer;
	timer.Start();

	// Reset Atomic Counter since we reconstruct every frame
	uint32_t usedNodeCount = hSVOLevelSizes.back() + svoLevelOffsets.CPUData().back();
	CUDA_CHECK(cudaMemset(dSVODense, 0xFF, sizeof(CSVONode) * (usedNodeCount + GI_DENSE_SIZE_CUBE)));
	CUDA_CHECK(cudaMemset(dSVOMaterial, 0x00, sizeof(CSVOMaterial) * (usedNodeCount + matSparseOffset)));

	dSVOLevelSizes.Memset(0x00, 0, dSVOLevelSizes.Size());
	std::fill(hSVOLevelSizes.begin(), hSVOLevelSizes.end(), 0);

	// Maxwell is faster with fully atomic code (CAS Locks etc.)
	// However kepler sucks(660ti) (100ms compared to 5ms) 
	if(CudaInit::CapabilityMajor() >= 5)
	{
		ConstructFullAtomic();
		AverageNodes(true);
	}
		
	else
	{
		ConstructLevelByLevel();
		AverageNodes(false);
	}
	
	//// DEBUG
	//GI_LOG("-------------------------------------------");
	//GI_LOG("Tree Node Data");
	//unsigned int i;
	//for(i = 0; i <= allocatorGrids[0]->depth - GI_DENSE_LEVEL + allocators.size() - 1; i++)
	//{
	//	if(i == 0) GI_LOG("#%d Dense : %d", GI_DENSE_LEVEL + i, GI_DENSE_SIZE_CUBE);
	//	else GI_LOG("#%d Level : %d", GI_DENSE_LEVEL + i, hSVOLevelSizes[i]);
	//}
	//unsigned int total = std::accumulate(hSVOLevelSizes.begin(),
	//									 hSVOLevelSizes.end(), 0);
	//GI_LOG("Total : %d", total);

	timer.Stop();
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoMaterialResource));
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoNodeResource));
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoLevelOffsetResource));

	CUDA_CHECK(cudaProfilerStop());
	return timer.ElapsedMilliS();
}

double GISparseVoxelOctree::AmbientOcclusionSurf(DeferredRenderer& dRenderer,
												 const Camera& camera,
												 float coneAngle,
												 float maxDistance,
												 float sampleDistanceRatio,
												 float intensityFactor)
{
	// Light Intensity Texture
	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glClearTexImage(liTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);

	// Update FrameTransform Matrices 
	// And its inverse realted buffer
	//assert(TraceWidth == DeferredRenderer::gBuffWidth);
	//assert(TraceHeight == DeferredRenderer::gBuffHeight);
	dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
	dRenderer.GetFTransform().Update(camera.generateTransform());

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	// Set Cascade Trace Data
	float3 pos = allocatorGrids[0]->position;
	uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
	uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
	svoTraceData.CPUData()[0] =
	{
		{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
		{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
		{
			static_cast<unsigned int>(allocators.size()),
			GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE,
			matSparseOffset,
			0
		}
	};
	svoTraceData.SendData();

	// Set Cone Trace Data
	svoConeParams.CPUData()[0] =
	{
		{maxDistance, std::tan(coneAngle), std::tan(coneAngle * 0.5f), sampleDistanceRatio},
		{intensityFactor, IEMath::Sqrt2, 0.0f, 0.0f}
	};
	svoConeParams.SendData();

	// Shaders
	computeAOSurf.Bind();

	// Buffers
	svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
	svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
	svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
	dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
	dRenderer.GetFTransform().Bind();
	svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);
	svoConeParams.BindAsUniformBuffer(U_CONE_PARAMS);

	// Images
	dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	dRenderer.GetGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
	glBindImageTexture(I_LIGHT_INENSITY, liTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);

	// Dispatch
	uint2 gridSize;
	gridSize.x = (TraceWidth * 4 + 32 - 1) / 32;
	gridSize.y = (TraceHeight + 8 - 1) / 8;
	glDispatchCompute(gridSize.x, gridSize.y, 1);

	// Render to window
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	dRenderer.ShowTexture(camera, liTexture);

	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);

	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
}

double GISparseVoxelOctree::AmbientOcclusion(DeferredRenderer& dRenderer,
											 const Camera& camera,
											 float coneAngle,
											 float maxDistance,
											 float sampleDistanceRatio)
{
	// Light Intensity Texture
	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glClearTexImage(liTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);

	// Update FrameTransform Matrices 
	// And its inverse realted buffer
	//assert(TraceWidth == DeferredRenderer::gBuffWidth);
	//assert(TraceHeight == DeferredRenderer::gBuffHeight);
	dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
	dRenderer.GetFTransform().Update(camera.generateTransform());

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	// Set Cascade Trace Data
	float3 pos = allocatorGrids[0]->position;
	uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
	uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
	svoTraceData.CPUData()[0] =
	{
		{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
		{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
		{
			static_cast<unsigned int>(allocators.size()),
			GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE,
			matSparseOffset,
			0
		}
	};
	svoTraceData.SendData();

	// Shaders
	computeAO.Bind();
	glUniform1f(U_CONE_ANGLE, coneAngle);
	glUniform1f(U_MAX_DISTANCE, maxDistance);
	glUniform1f(U_SAMPLE_DISTANCE, sampleDistanceRatio);

	// Buffers
	svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
	svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
	svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
	dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
	dRenderer.GetFTransform().Bind();
	svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);

	// Images
	dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	dRenderer.GetGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
	glBindImageTexture(I_LIGHT_INENSITY, liTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);

	// Dispatch
	uint2 gridSize;
	gridSize.x = (TraceWidth * 4 + 16 - 1) / 16;
	gridSize.y = (TraceHeight + 16 - 1) / 16;
	glDispatchCompute(gridSize.x, gridSize.y, 1);

	// Render to window
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	dRenderer.ShowTexture(camera, liTexture);

	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);

	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
}

double GISparseVoxelOctree::DebugDeferredSVO(DeferredRenderer& dRenderer,
											 const Camera& camera,
											 uint32_t renderLevel,
											 SVOTraceType type)
{
	// Update FrameTransform Matrices 
	// And its inverse realted buffer
	assert(TraceWidth == DeferredRenderer::gBuffWidth);
	assert(TraceHeight == DeferredRenderer::gBuffHeight);
	dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
	dRenderer.GetFTransform().Update(camera.generateTransform());

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	// Set Cascade Trace Data
	float3 pos = allocatorGrids[0]->position;
	uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
	uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
	svoTraceData.CPUData()[0] =
	{
		{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
		{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
		{
			static_cast<unsigned int>(allocators.size()),
			GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE,
			matSparseOffset,
			renderLevel
		}
	};
	svoTraceData.SendData();

	// Shaders
	computeVoxTraceDeferred.Bind();
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(type));

	// Buffers
	svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
	svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
	svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
	dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
	dRenderer.GetFTransform().Bind();
	svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);

	// Images
	dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	glBindImageTexture(I_COLOR_FB, liTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);

	// Dispatch
	uint2 gridSize;
	gridSize.x = (TraceWidth + 16 - 1) / 16;
	gridSize.y = (TraceHeight + 16 - 1) / 16;
	glDispatchCompute(gridSize.x, gridSize.y, 1);

	// Render to window
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	dRenderer.ShowTexture(camera, liTexture);

	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);

	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
}

double GISparseVoxelOctree::DebugTraceSVO(DeferredRenderer& dRenderer,
										  const Camera& camera,
										  uint32_t renderLevel,
										  SVOTraceType type)
{
	// Update FrameTransform Matrices 
	// And its inverse realted buffer
	dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
	dRenderer.GetFTransform().Update(camera.generateTransform());

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// Set Cascade Trace Data
	float3 pos = allocatorGrids[0]->position;
	uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
	uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
	svoTraceData.CPUData()[0] = 
	{
		{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
		{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
		{
			static_cast<unsigned int>(allocators.size()), 
			GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE,
			matSparseOffset,
			renderLevel
		}
	};
	svoTraceData.SendData();

	// Shaders
	computeVoxTraceWorld.Bind();
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(type));

	// Buffers
	svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
	svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
	svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
	dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
	dRenderer.GetFTransform().Bind();
	svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);

	// Images
	glBindImageTexture(I_COLOR_FB, liTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);

	// Dispatch
	uint2 gridSize;
	gridSize.x = (TraceWidth + 16 - 1) / 16;
	gridSize.y = (TraceHeight + 16 - 1) / 16;
	glDispatchCompute(gridSize.x, gridSize.y, 1);
	
	// Render to window
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	dRenderer.ShowTexture(camera, liTexture);

	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
	
	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
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

const CSVOConstants& GISparseVoxelOctree::SVOConsts() const
{
	return hSVOConstants;
}