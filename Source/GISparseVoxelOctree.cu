#include "GISparseVoxelOctree.h"
#include "SceneI.h"
#include "Globals.h"
#include "SceneLights.h"
#include "DeferredRenderer.h"
#include "GLSLBindPoints.h"
#include "GIVoxelPages.h"
#include "GIVoxelCache.h"
#include "SVOKernels.cuh"
#include "CudaTimer.h"
#include <numeric>
#include <cuda_gl_interop.h>

//#include "SVOKernels.cuh"
//#include "CudaTimer.h"
//#include "Macros.h"
//#include "Camera.h"
//#include "Globals.h"
//#include "CDebug.cuh"
//#include "IEUtility/IEMath.h"

//#include "GLSLBindPoints.h"

GISparseVoxelOctree::ShadowMapsCUDA::ShadowMapsCUDA()
	: lightCount(0)
	, matrixOffset(0)
	, lightOffset(0)
	, shadowMapResource(nullptr)
	, lightBufferResource(nullptr)
	, shadowMapArray(nullptr)
	, tShadowMapArray(0)
	, dLightParamArray(nullptr)
	, dLightVPMatrixArray(nullptr)
{}

GISparseVoxelOctree::ShadowMapsCUDA::ShadowMapsCUDA(const SceneLights& sLights)
	: lightCount(sLights.getLightCount())
	, matrixOffset(sLights.getMatrixOffset())
	, lightOffset(sLights.getLightOffset())
	, shadowMapResource(nullptr)
	, lightBufferResource(nullptr)
	, shadowMapArray(nullptr)
	, tShadowMapArray(0)
	, dLightParamArray(nullptr)
	, dLightVPMatrixArray(nullptr)
{
	// Here i gurantee that these buffers are read only, thus const_cast
	GLuint glBuffer = const_cast<SceneLights&>(sLights).getGLBuffer();
	GLuint shadowMaps = const_cast<SceneLights&>(sLights).getShadowTextureArrayView();

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&lightBufferResource, glBuffer,
											cudaGraphicsMapFlagsReadOnly));
	CUDA_CHECK(cudaGraphicsGLRegisterImage(&shadowMapResource, shadowMaps,
										   GL_TEXTURE_2D_ARRAY,
										   cudaGraphicsMapFlagsReadOnly));
}

GISparseVoxelOctree::ShadowMapsCUDA::ShadowMapsCUDA(ShadowMapsCUDA&& other)
	: lightCount(other.lightCount)
	, matrixOffset(other.matrixOffset)
	, lightOffset(other.lightOffset)
	, shadowMapResource(other.shadowMapResource)
	, lightBufferResource(other.lightBufferResource)
	, shadowMapArray(other.shadowMapArray)
	, tShadowMapArray(other.tShadowMapArray)
	, dLightParamArray(other.dLightParamArray)
	, dLightVPMatrixArray(other.dLightVPMatrixArray)
{
	other.shadowMapResource = nullptr;
	other.lightBufferResource = nullptr;
}

GISparseVoxelOctree::ShadowMapsCUDA& GISparseVoxelOctree::ShadowMapsCUDA::operator=(ShadowMapsCUDA&& other)
{
	assert(&other != this);
	if(lightBufferResource)
		CUDA_CHECK(cudaGraphicsUnregisterResource(lightBufferResource));
	if(shadowMapResource)
		CUDA_CHECK(cudaGraphicsUnregisterResource(shadowMapResource));

	lightCount = other.lightCount;
	matrixOffset = other.matrixOffset;
	lightOffset = other.lightOffset;
	shadowMapResource = other.shadowMapResource;
	lightBufferResource = other.lightBufferResource;
	shadowMapArray = other.shadowMapArray;
	tShadowMapArray = other.tShadowMapArray;
	dLightParamArray = other.dLightParamArray;
	dLightVPMatrixArray = other.dLightVPMatrixArray;

	other.shadowMapResource = nullptr;
	other.lightBufferResource = nullptr;
	return *this;
}

GISparseVoxelOctree::ShadowMapsCUDA::~ShadowMapsCUDA()
{
	if(lightBufferResource)
		CUDA_CHECK(cudaGraphicsUnregisterResource(lightBufferResource));
	if(shadowMapResource)
		CUDA_CHECK(cudaGraphicsUnregisterResource(shadowMapResource));
}

void GISparseVoxelOctree::ShadowMapsCUDA::Map()
{
	assert(dLightParamArray == nullptr);
	assert(dLightVPMatrixArray == nullptr);
	assert(shadowMapArray == nullptr);
	assert(tShadowMapArray == 0);

	CUDA_CHECK(cudaGraphicsMapResources(1, &shadowMapResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&shadowMapArray, shadowMapResource));

	// Texture Generation From Cuda Array
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};
	resDesc.resType = cudaResourceTypeMipmappedArray;
	resDesc.res.mipmap.mipmap = shadowMapArray;

	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.addressMode[2] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;

	CUDA_CHECK(cudaCreateTextureObject(&tShadowMapArray, &resDesc, &texDesc, nullptr));

	// Buffer
	size_t size;
	uint8_t* glBufferCUDA = nullptr;
	CUDA_CHECK(cudaGraphicsMapResources(1, &lightBufferResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glBufferCUDA),
													&size, lightBufferResource));

	dLightParamArray = reinterpret_cast<const CLight*>(glBufferCUDA + lightOffset);
	dLightVPMatrixArray = reinterpret_cast<const CMatrix4x4*>(glBufferCUDA + matrixOffset);
}

void GISparseVoxelOctree::ShadowMapsCUDA::Unmap()
{
	assert(dLightParamArray != nullptr);
	assert(dLightVPMatrixArray != nullptr);
	assert(shadowMapArray != nullptr);
	assert(tShadowMapArray != 0);

	// Unmap Texture
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &shadowMapResource));
	CUDA_CHECK(cudaDestroyTextureObject(tShadowMapArray));
	tShadowMapArray = 0;
	shadowMapArray = nullptr;

	// Unmap Buffer
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &lightBufferResource));
	dLightParamArray = nullptr;
	dLightVPMatrixArray = nullptr;
}

const CLight* GISparseVoxelOctree::ShadowMapsCUDA::LightParamArray() const
{
	return dLightParamArray;
}

const CMatrix4x4* GISparseVoxelOctree::ShadowMapsCUDA::LightVPMatrices() const
{
	return dLightVPMatrixArray;
}

cudaTextureObject_t GISparseVoxelOctree::ShadowMapsCUDA::ShadowMapArray() const
{
	return tShadowMapArray;
}

uint32_t GISparseVoxelOctree::ShadowMapsCUDA::LightCount() const
{
	return lightCount;
}

GISparseVoxelOctree::GISparseVoxelOctree()
	: octreeParams(nullptr)
	, scene(nullptr)
	, octreeUniformsOffset(0)
	, indirectUniformsOffset(0)
	, illumOffsetsOffset(0)
	, nodeOffset(0)
	, illumOffset(0)
	, gpuResource(nullptr)
	, dLevelCapacities(nullptr)
	, dLevelSizes(nullptr)
	, dOctreeLevels(nullptr)
	, nodeIllumDifference(0)
{}

GISparseVoxelOctree::GISparseVoxelOctree(const OctreeParameters& octreeParams,
										 const SceneI* currentScene,
										 const size_t sizes[])
	: octreeParams(&octreeParams)
	, scene(currentScene)
	, octreeUniformsOffset(0)
	, indirectUniformsOffset(0)
	, illumOffsetsOffset(0)
	, nodeOffset(0)
	, illumOffset(0)
	, gpuResource(nullptr)
	, dLevelCapacities(nullptr)
	, dLevelSizes(nullptr)
	, dOctreeLevels(nullptr)
	, nodeIllumDifference(0)
	, shadowMaps(currentScene->getSceneLights())
	, compVoxTraceWorld(ShaderType::COMPUTE, "Shaders/VoxTraceWorld.comp")
	, compVoxSampleWorld(ShaderType::COMPUTE, "Shaders/VoxTraceDeferred.comp")
	, compGI(ShaderType::COMPUTE, "Shaders/VoxGI.comp")
{	
	// Generate Initial Sizes for each level
	std::vector<uint32_t> levelCapacities(octreeParams.MaxSVOLevel + 1, 0);
	std::vector<uint32_t> internalOffsets(octreeParams.MaxSVOLevel + 1, 0);
	size_t offset = 0;
	for(uint32_t i = octreeParams.MinSVOLevel; i < octreeParams.MaxSVOLevel + 1; i++)
	{
		if(i == octreeParams.DenseLevel) nodeIllumDifference = offset;
		size_t levelSize = (i <= octreeParams.DenseLevel)
								? ((1 << i) * (1 << i) * (1 << i))
								: sizes[i];
		internalOffsets[i] = static_cast<uint32_t>(offset);
		levelCapacities[i] = static_cast<uint32_t>(levelSize);
		offset += levelSize;
	}
	size_t totalIllumSize = offset;
	size_t totalNodeSize = offset - nodeIllumDifference;
	hIllumOffsets = internalOffsets;

	// Allocation of OpenGL Side
	offset = 0;
	// OctreeUniforms
	octreeUniformsOffset = offset;
	offset += sizeof(OctreeUniforms);
	// IndirectUniforms	
	offset = DeviceOGLParameters::UBOAlignOffset(offset);
	indirectUniformsOffset = offset;
	offset += sizeof(IndirectUniforms);
	// IllumOffsets
	offset = DeviceOGLParameters::SSBOAlignOffset(offset);
	illumOffsetsOffset = offset;
	offset += (octreeParams.MaxSVOLevel + 1) * sizeof(uint32_t);
	// Nodes
	offset = DeviceOGLParameters::SSBOAlignOffset(offset);
	nodeOffset = offset;
	offset += (totalNodeSize) * sizeof(CSVONode);
	// Illum Data
	offset = DeviceOGLParameters::SSBOAlignOffset(offset);
	illumOffset = offset;
	offset += (totalIllumSize) * sizeof(CSVOIllumination);
	
	// Offsets Generated Allocate
	oglData.Resize(offset, false);

	// Now CUDA
	size_t totalSize = (octreeParams.MaxSVOLevel + 1) * (sizeof(uint32_t) * 2 + 
														 sizeof(CSVOLevel));
	cudaData.Resize(totalSize);

	// Allocation of CUDA Side
	offset = 0;
	// Level Capacities
	dLevelCapacities = reinterpret_cast<uint32_t*>(cudaData.Data() + offset);
	offset += (sizeof(uint32_t) * (octreeParams.MaxSVOLevel + 1));
	// Level Sizes
	dLevelSizes = reinterpret_cast<uint32_t*>(cudaData.Data() + offset);
	offset += (sizeof(uint32_t) * (octreeParams.MaxSVOLevel + 1));
	// Octree Levels
	dOctreeLevels = reinterpret_cast<CSVOLevel*>(cudaData.Data() + offset);
	offset += (sizeof(CSVOLevel) * (octreeParams.MaxSVOLevel + 1));
	assert(offset == totalSize);

	// Load Level Capacities
	CUDA_CHECK(cudaMemcpy(const_cast<uint32_t*>(dLevelCapacities),
						  levelCapacities.data(),
						  (octreeParams.MaxSVOLevel + 1) * sizeof(uint32_t),
						  cudaMemcpyHostToDevice));

	// Register CUDA Resource
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&gpuResource, oglData.getGLBuffer(),
											cudaGraphicsMapFlagsNone));
}

GISparseVoxelOctree::GISparseVoxelOctree(GISparseVoxelOctree&& other)
	: octreeParams(other.octreeParams)
	, scene(other.scene)
	, oglData(std::move(other.oglData))
	, octreeUniformsOffset(other.octreeUniformsOffset)
	, indirectUniformsOffset(other.indirectUniformsOffset)
	, illumOffsetsOffset(other.illumOffsetsOffset)
	, nodeOffset(other.nodeOffset)
	, illumOffset(other.illumOffset)
	, gpuResource(other.gpuResource)
	, cudaData(std::move(other.cudaData))
	, dLevelCapacities(other.dLevelCapacities)
	, dLevelSizes(other.dLevelSizes)
	, dOctreeLevels(other.dOctreeLevels)
	, hIllumOffsets(std::move(other.hIllumOffsets))
	, nodeIllumDifference(other.nodeIllumDifference)
	, shadowMaps(std::move(other.shadowMaps))
	, compVoxTraceWorld(std::move(other.compVoxTraceWorld))
	, compVoxSampleWorld(std::move(other.compVoxSampleWorld))
	, compGI(std::move(other.compGI))
{
	other.gpuResource = nullptr;
}

GISparseVoxelOctree& GISparseVoxelOctree::operator=(GISparseVoxelOctree&& other)
{
	assert(&other != this);
	if(gpuResource) CUDA_CHECK(cudaGraphicsUnregisterResource(gpuResource));

	octreeParams = other.octreeParams;
	scene = other.scene;
	oglData = std::move(other.oglData);
	octreeUniformsOffset = other.octreeUniformsOffset;
	indirectUniformsOffset = other.indirectUniformsOffset;
	illumOffsetsOffset = other.illumOffsetsOffset;
	nodeOffset = other.nodeOffset;
	illumOffset = other.illumOffset;
	gpuResource = other.gpuResource;
	cudaData = std::move(other.cudaData);
	dLevelCapacities = other.dLevelCapacities;
	dLevelSizes = other.dLevelSizes;
	dOctreeLevels = other.dOctreeLevels;
	hIllumOffsets = std::move(other.hIllumOffsets);
	nodeIllumDifference = other.nodeIllumDifference;
	shadowMaps = std::move(other.shadowMaps);
	compVoxTraceWorld = std::move(other.compVoxTraceWorld);
	compVoxSampleWorld = std::move(other.compVoxSampleWorld);
	compGI = std::move(other.compGI);

	other.gpuResource = nullptr;
	return *this;
}

GISparseVoxelOctree::~GISparseVoxelOctree()
{
	if(gpuResource) CUDA_CHECK(cudaGraphicsUnregisterResource(gpuResource));
}

void GISparseVoxelOctree::MapOGLData()
{
	// Get Node Pointer
	CUDA_CHECK(cudaGraphicsMapResources(1, &gpuResource));
	size_t size; uint8_t* oglCudaPtr;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&oglCudaPtr),
													&size, gpuResource));
	assert(size == oglData.Capacity());
	
	// Recieve Used Pointer Sizes
	std::vector<uint32_t> levelAllocators(octreeParams->MaxSVOLevel + 1);
	CUDA_CHECK(cudaMemcpy(levelAllocators.data(), dLevelSizes,
						  (octreeParams->MaxSVOLevel + 1) * sizeof(uint32_t),
						  cudaMemcpyDeviceToHost));

	std::vector<CSVOLevel> svoLevels(octreeParams->MaxSVOLevel + 1, {nullptr, nullptr});
	for(uint32_t i = octreeParams->MinSVOLevel; i < octreeParams->MaxSVOLevel + 1; i++)
	{
		CSVONode* nodePtr = nullptr;
		CSVOIllumination* illumPtr = reinterpret_cast<CSVOIllumination*>(oglCudaPtr + illumOffset)
																		 + hIllumOffsets[i];
		if(i < octreeParams->DenseLevel)
		{
			nodePtr = reinterpret_cast<CSVONode*>(oglCudaPtr + nodeOffset) 
												  + (hIllumOffsets[i] - nodeIllumDifference);

			// Clear used node pointers
			//CUDA_CHECK(cudaMemset(nodePtr, 0xFF, levelAllocators[i]));
		}

		svoLevels[i].gLevelNodes = nodePtr;
		svoLevels[i].gLevelIllum = illumPtr;

		// Clear used illum
		//CUDA_CHECK(cudaMemset(illumPtr, 0x00, levelAllocators[i]));
	}

	// Clear level allocators
	CUDA_CHECK(cudaMemset(dLevelSizes, 0x00, octreeParams->MaxSVOLevel + 1));

	// Copy Generated Pointers
	CUDA_CHECK(cudaMemcpy(dOctreeLevels, svoLevels.data(),
						  (octreeParams->MaxSVOLevel + 1) * sizeof(CSVOLevel),
						  cudaMemcpyHostToDevice));
}

void GISparseVoxelOctree::UnmapOGLData()
{
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &gpuResource));
}

double GISparseVoxelOctree::GenerateHierarchy(bool doTiming,
											  // Page System
											  const GIVoxelPages& pages,
											  // Cache System
											  const GIVoxelCache& caches,
											  // Constants
											  uint32_t batchCount,
											  const LightInjectParameters& injectParams,
											  bool injectOn)
{
	CudaTimer t;
	if(doTiming) t.Start();

	// Gen LI Params
	CLightInjectParameters liParams = 
	{
		injectOn,
		injectParams.camPos,
		injectParams.camDir,

		shadowMaps.LightVPMatrices(),
		shadowMaps.LightParamArray(),

		injectParams.depthNear,
		injectParams.depthFar,
		shadowMaps.ShadowMapArray(),
		shadowMaps.LightCount()
	};
	
	//// KC
	//int gridSize = 1;// CudaInit::GenBlockSize(static_cast<int>(pages.PageCount() * GIVoxelPages::PageSize));
	//int blockSize = CudaInit::TBP;
	//SVOReconstruct<<<gridSize, blockSize>>>(// SVO
	//									    dOctreeLevels,
	//									    reinterpret_cast<const CSVOLevelConst*>(dOctreeLevels),
	//									    dLevelSizes,
	//									    dLevelCapacities,
	//									    // Voxel Pages
	//									    pages.getVoxelPages(),
	//									    pages.getVoxelGrids(),
	//									    // Cache Data (for Voxel Albedo)
	//									    caches.getDeviceCascadePointersDevice().Data(),
	//									    // Light Injection Related
	//									    liParams,
	//									    // Limits
	//									    *octreeParams,
	//									    batchCount);
	//CUDA_KERNEL_CHECK();
	//cudaDeviceSynchronize();
	//GI_LOG("-----");

	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

double GISparseVoxelOctree::AverageNodes(bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();


	// Work

	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

void GISparseVoxelOctree::UpdateSVO(// Timing Related
									double& reconstructTime,
									double& averageTime,
									bool doTiming,
									// Page System
									const GIVoxelPages& pages,
									// Cache System
									const GIVoxelCache& caches,
									// Constants
									uint32_t batchCount,
									const LightInjectParameters& injectParams,
									bool injectOn)
{
	MapOGLData();
	shadowMaps.Map();

	reconstructTime = GenerateHierarchy(doTiming,
										pages, caches, batchCount,
										injectParams, injectOn);
	averageTime = AverageNodes(doTiming);

	shadowMaps.Unmap();
	UnmapOGLData();
}

void GISparseVoxelOctree::UpdateOctreeUniforms(const IEVector3& outerCascadePos)
{
	// Octree Uniforms
	OctreeUniforms u = {};
	u.worldPos = outerCascadePos;
	u.baseSpan = octreeParams->BaseSpan;
	u.gridSize = octreeParams->CascadeBaseLevelSize;
	u.minSVOLevel = octreeParams->MinSVOLevel;
	u.maxSVOLevel = octreeParams->MaxSVOLevel;
	u.denseLevel = octreeParams->DenseLevel;
	u.cascadeCount = octreeParams->CascadeCount;
	u.nodeOffsetDifference = static_cast<uint32_t>(nodeIllumDifference);
	std::memcpy(oglData.CPUData().data() + octreeUniformsOffset,
				&u, sizeof(OctreeUniforms));
	oglData.SendSubData(static_cast<uint32_t>(octreeUniformsOffset), sizeof(OctreeUniforms));
}

void GISparseVoxelOctree::UpdateIndirectUniforms(const IndirectUniforms& indirectUniforms)
{
	std::memcpy(oglData.CPUData().data() + indirectUniformsOffset,
				&indirectUniforms,
				sizeof(IndirectUniforms));
	oglData.SendSubData(static_cast<uint32_t>(indirectUniformsOffset), sizeof(IndirectUniforms));
}

double GISparseVoxelOctree::GlobalIllumination(GLuint outputTexture,
											   const DeferredRenderer& dRenderer,
											   const Camera& camera,
											   const IndirectUniforms&,
											   bool giOn,
											   bool aoOn,
											   bool specularOn)
{
	// Light Intensity Texture
	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glClearTexImage(outputTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);
	
	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);
		
	// Shaders
	compGI.Bind();
		
	// Uniforms
	glUniform1ui(U_CAST_SPECULAR_CONE, (specularOn) ? 1u : 0u);
	
	// Uniform Buffers
	// Frame transform already bound
	dRenderer.BindInvFrameTransform(U_INVFTRANSFORM);
	oglData.BindAsUniformBuffer(U_OCTREE_UNIFORMS, static_cast<uint32_t>(octreeUniformsOffset), 
								sizeof(OctreeUniforms));
	oglData.BindAsUniformBuffer(U_INDIRECT_UNIFORMS, static_cast<uint32_t>(indirectUniformsOffset), 
								sizeof(IndirectUniforms));

	// SSBO Buffers
	oglData.BindAsUniformBuffer(LU_SVO_LEVEL_OFFSET, 
								static_cast<uint32_t>(illumOffsetsOffset), 
								sizeof(uint32_t) * (octreeParams->MaxSVOLevel + 1));
	oglData.BindAsUniformBuffer(LU_SVO_NODE, static_cast<uint32_t>(nodeOffset), 
								static_cast<uint32_t>(illumOffset - nodeOffset));
	oglData.BindAsUniformBuffer(LU_SVO_ILLUM, static_cast<uint32_t>(illumOffset), 
								static_cast<uint32_t>(oglData.Count() - illumOffset));

	// Textures
	dRenderer.getGBuffer().BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	dRenderer.getGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	dRenderer.getGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);

	// Images
	glBindImageTexture(I_OUT_TEXTURE, outputTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
	
	// Dispatch
	uint2 gridSize;
	gridSize.x = (TraceWidth + 16 - 1) / 16;
	gridSize.y = (TraceHeight + 16 - 1) / 16;
	glDispatchCompute(gridSize.x, gridSize.y, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	
	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
	
	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
}

double GISparseVoxelOctree::DebugTraceSVO(GLuint outputTexture,
										  const DeferredRenderer&,
										  const Camera& camera,
										  uint32_t renderLevel,
										  OctreeRenderType)
{
	return 0.0f;
}

double GISparseVoxelOctree::DebugSampleSVO(GLuint& outputTexture,
										   const DeferredRenderer&,
										   const Camera& camera,
										   uint32_t renderLevel,
										   OctreeRenderType)
{
	return 0.0f;
}

size_t GISparseVoxelOctree::MemoryUsage() const
{
	return oglData.Capacity() + cudaData.Size();
}

//GISparseVoxelOctree::GISparseVoxelOctree(const OctreeParameters& octreeParams)
//	: octreeParams(octreeParams)
//	, dSVOConstants(1)
//	, computeVoxTraceWorld(ShaderType::COMPUTE, "Shaders/VoxTraceWorld.comp")
//	, computeVoxTraceDeferredLerp(ShaderType::COMPUTE, "Shaders/VoxTraceDeferredLerp.comp")
//	, computeVoxTraceDeferred(ShaderType::COMPUTE, "Shaders/VoxTraceDeferred.comp")
//	, computeAO(ShaderType::COMPUTE, "Shaders/VoxAO.comp")
//	, computeGI(ShaderType::COMPUTE, "Shaders/VoxGI.comp")
//	, computeGauss32(ShaderType::COMPUTE, "Shaders/Gauss32.comp")
//	, computeEdge(ShaderType::COMPUTE, "Shaders/EdgeDetect.comp")
//	, computeAOSurf(ShaderType::COMPUTE, "Shaders/SurfAO.comp")
//	, computeLIApply(ShaderType::COMPUTE, "Shaders/ApplyVoxLI.comp")
//	, svoTraceData(1)
//	, svoConeParams(1)
//	, svoNodeResource(nullptr)
//	, svoLevelOffsetResource(nullptr)
//	, svoMaterialResource(nullptr)
//	, svoDenseNodeResource(nullptr)
//	, sceneShadowMapResource(nullptr)
//	, sceneLightParamResource(nullptr)
//	, sceneVPMatrixResource(nullptr)
//	, tSVODenseNode(0)
//	, sSVODenseNode(0)
//	, tShadowMapArray(0)
//	, dSVODenseNodeArray(nullptr)
//	, traceTexture(0)
//	, gaussTex(0)
//	, edgeTex(0)
//	, svoDenseMat(0)
//	, sSVODenseMat(octreeParams.DenseLevelCount, 0)
//	, dSVODenseMatArray(octreeParams.DenseLevelCount, nullptr)
//	, nodeSampler(0)
//	, materialSampler(0)
//	, gaussSampler(0)
//{
//	svoTraceData.AddData({});
//	svoConeParams.AddData({});
//
//	// Light Intensity Tex
//	glGenTextures(1, &traceTexture);
//	glBindTexture(GL_TEXTURE_2D, traceTexture);
//	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8/*GL_RGBA16*/, TraceWidth, TraceHeight);
//
//	// Gauss Intermediary Tex
//	glGenTextures(1, &gaussTex);
//	glBindTexture(GL_TEXTURE_2D, gaussTex);
//	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8/*GL_RGBA16*/, TraceWidth, TraceHeight);
//
//	// Edge Map Tex
//	glGenTextures(1, &edgeTex);
//	glBindTexture(GL_TEXTURE_2D, edgeTex);
//	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG8, TraceWidth, TraceHeight);
//	
//	//// Dense Tex
//	//glGenTextures(1, &svoDenseNode);
//	//glBindTexture(GL_TEXTURE_3D, svoDenseNode);
//	//glTexStorage3D(GL_TEXTURE_3D, 1, GL_R32UI, GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE);
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&svoDenseNodeResource, svoDenseNode, GL_TEXTURE_3D, 
//	//									   cudaGraphicsRegisterFlagsSurfaceLoadStore)); //|
//	//									   //cudaGraphicsRegisterFlagsWriteDiscard*/));
//
//	//// Mat Texture Binds
//	//// Mipped 3D tex
//	//glGenTextures(1, &svoDenseMat);	
//	//glBindTexture(GL_TEXTURE_3D, svoDenseMat);
//	//glTexStorage3D(GL_TEXTURE_3D, GI_DENSE_TEX_COUNT, GL_RGBA32UI, GI_DENSE_SIZE, GI_DENSE_SIZE, GI_DENSE_SIZE);
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&svoDenseTexResource, svoDenseMat, GL_TEXTURE_3D,
//	//									   cudaGraphicsRegisterFlagsSurfaceLoadStore)); //|
//	//									   //cudaGraphicsRegisterFlagsWriteDiscard));
//	
//	// Flat Sampler for Node Index Fetch
//	glGenSamplers(1, &nodeSampler);
//	glSamplerParameteri(nodeSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glSamplerParameteri(nodeSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
//	glSamplerParameteri(nodeSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//	glSamplerParameteri(nodeSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//	glSamplerParameteri(nodeSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//
//	// Nearest Sample for Material Fetch since its interger tex no interpolation
//	glGenSamplers(1, &materialSampler);
//	glSamplerParameteri(materialSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
//	glSamplerParameteri(materialSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
//	glSamplerParameteri(materialSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
//	glSamplerParameteri(materialSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
//	glSamplerParameteri(materialSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
//
//	// Bilinear Sample for Gauss Fetch (Out of bounds are zero)
//	GLfloat col[] = {0.0f, 0.0f, 0.0f, 0.0f};
//	glGenSamplers(1, &gaussSampler);
//	glSamplerParameteri(gaussSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
//	glSamplerParameteri(gaussSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
//	glSamplerParameteri(gaussSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
//	glSamplerParameteri(gaussSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
//	glSamplerParameteri(gaussSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
//	glSamplerParameterfv(gaussSampler, GL_TEXTURE_BORDER_COLOR, col);
//}
//
//GISparseVoxelOctree::~GISparseVoxelOctree()
//{
//	if(svoNodeResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoNodeResource));
//	if(svoMaterialResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoMaterialResource));
//	if(svoLevelOffsetResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoLevelOffsetResource));
//	if(svoDenseTexResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoDenseTexResource));
//	for(unsigned int i = 0; i < octreeParams.DenseLevelCount; i++)
//	{
//		if(sSVODenseMat[i]) CUDA_CHECK(cudaDestroySurfaceObject(sSVODenseMat[i]));
//	}
//	if(svoDenseNodeResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoDenseNodeResource));
//	if(tSVODenseNode) CUDA_CHECK(cudaDestroyTextureObject(tSVODenseNode));
//	if(sSVODenseNode) CUDA_CHECK(cudaDestroySurfaceObject(sSVODenseNode));
//
//	if(traceTexture) glDeleteTextures(1, &traceTexture);
//	if(gaussTex) glDeleteTextures(1, &gaussTex);
//	if(edgeTex) glDeleteTextures(1, &edgeTex);
//	if(svoDenseNode) glDeleteTextures(1, &svoDenseNode);
//	if(svoDenseMat) glDeleteTextures(1, &svoDenseMat);
//	if(nodeSampler) glDeleteSamplers(1, &nodeSampler);
//	if(materialSampler) glDeleteSamplers(1, &materialSampler);
//	if(materialSampler) glDeleteSamplers(1, &gaussSampler);
//}

//void GISparseVoxelOctree::LinkAllocators(std::vector<GICudaAllocator*> newAllocators,
//										 const uint32_t levelCounts[])
//{
//	//allocatorGrids.clear();
//	//allocators.resize(newAllocators.size());
//	//allocatorGrids.resize(newAllocators.size());
//
//	//assert(newAllocators.size() > 0);
//
//	//std::copy(newAllocators.data(), newAllocators.data() + newAllocators.size(), allocators.data());
//	//for(unsigned int i = 0; i < newAllocators.size(); i++)
//	//	allocatorGrids[i] = &(newAllocators[i]->GetVoxelGridHost());
//
//	//size_t sparseNodeCount = allocatorGrids[0]->depth + newAllocators.size() - GI_DENSE_LEVEL;
//	//uint32_t totalLevel = allocatorGrids[0]->depth + static_cast<uint32_t>(newAllocators.size() - 1);
//
// //   size_t totalAlloc = 0;
// //   for(unsigned int i = GI_DENSE_LEVEL + 1; i <= totalLevel; i++)
// //   {
// //       totalAlloc += levelCounts[i];
// //   }
//
//	//// TODO: More Dynamic Allocation Scheme
//	//hSVOLevelTotalSizes.resize(sparseNodeCount);
//	//dSVOLevelTotalSizes.Resize(sparseNodeCount);
//	//dSVOLevelSizes.Resize(sparseNodeCount);
//	//hSVOLevelSizes.resize(sparseNodeCount);
//	////svoLevelOffsets.Resize(sparseNodeCount);
//
//	////// Sparse Portion
//	////svoNodeBuffer.Resize(totalAlloc + DenseSizeCube);
//	////svoMaterialBuffer.Resize(totalAlloc);
//	//
//	//// Register
//	//if(svoNodeResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoNodeResource));
//	//if(svoMaterialResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoMaterialResource));
//	//if(svoLevelOffsetResource) CUDA_CHECK(cudaGraphicsUnregisterResource(svoLevelOffsetResource));
//	//CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoNodeResource, 
//	//										svoNodeBuffer.getGLBuffer(), 
//	//										cudaGLMapFlagsWriteDiscard));
//	//CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoMaterialResource, 
//	//										svoMaterialBuffer.getGLBuffer(), 
//	//										cudaGLMapFlagsWriteDiscard));
//	//CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&svoLevelOffsetResource,
//	//										svoLevelOffsets.getGLBuffer(),
//	//										cudaGLMapFlagsReadOnly));
//
//	//// Actual Data Init
//	//GLuint allOne = 0xFFFFFFFF;
//	//GLuint zero = 0;
//
//	//glBindBuffer(GL_COPY_WRITE_BUFFER, svoNodeBuffer.getGLBuffer());
//	//glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &allOne);
//
//	//glBindBuffer(GL_COPY_WRITE_BUFFER, svoMaterialBuffer.getGLBuffer());
//	//glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &zero);
//
//	//dSVOLevelSizes.Memset(0x00, 0, dSVOLevelSizes.Size());
//	//std::fill(hSVOLevelSizes.begin(), hSVOLevelSizes.end(), 0);
//	//std::copy(levelCounts + GI_DENSE_LEVEL, 
//	//		  levelCounts + GI_DENSE_LEVEL + sparseNodeCount, 
//	//		  hSVOLevelTotalSizes.data());
//	//hSVOLevelTotalSizes[0] = DenseSizeCube;
//	//dSVOLevelTotalSizes = hSVOLevelTotalSizes;
//
//	//// SVO Constants set
//	//hSVOConstants.denseDepth = DenseLevel;
//	//hSVOConstants.denseDim = DenseSize;
//	//hSVOConstants.totalDepth = totalLevel;
//	//hSVOConstants.numCascades = static_cast<uint32_t>(newAllocators.size());
//
//	//// Offset Set
//	//uint32_t levelOffset = 0;
//	//svoLevelOffsets.CPUData().clear();
//	//for(unsigned int i = GI_DENSE_LEVEL; i <= totalLevel; i++)
//	//{
//	//	svoLevelOffsets.AddData(levelOffset);
//	//	levelOffset += (i != GI_DENSE_LEVEL) ? levelCounts[i] : 0;
//	//}
//	//svoLevelOffsets.SendData();
//	//assert(levelOffset <= totalAlloc);
//
//	//// Copy to device
//	//CUDA_CHECK(cudaMemcpy(dSVOConstants.Data(), 
//	//					  &hSVOConstants, 
//	//					  sizeof(CSVOConstants), 
//	//					  cudaMemcpyHostToDevice));
//}
//
//void GISparseVoxelOctree::LinkSceneShadowMaps(SceneI* scene)
//{
//	//GLuint lightParamBuffer = scene->getSceneLights().getGLBuffer();
//	//GLuint shadowMapTexture = scene->getSceneLights().getShadowArrayGL();
//
//
//	//if(sceneShadowMapResource) CUDA_CHECK(cudaGraphicsUnregisterResource(sceneShadowMapResource));
//	//if(sceneLightParamResource) CUDA_CHECK(cudaGraphicsUnregisterResource(sceneLightParamResource));
//	//if(sceneVPMatrixResource) CUDA_CHECK(cudaGraphicsUnregisterResource(sceneVPMatrixResource));
//	//CUDA_CHECK(cudaGraphicsGLRegisterImage(&sceneShadowMapResource, shadowMapTexture, GL_TEXTURE_2D_ARRAY,
//	//									   cudaGraphicsRegisterFlagsReadOnly));
//	//CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&sceneLightParamResource, lightParamBuffer,
//	//										cudaGraphicsRegisterFlagsReadOnly));
//	//CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&sceneVPMatrixResource, lightVPBuffer,
//	//										cudaGraphicsRegisterFlagsReadOnly));
//}
//
//void GISparseVoxelOctree::CreateSurfFromArray(cudaArray_t& arr, cudaSurfaceObject_t& surf)
//{
//	// Texture of SVO Dense
//	cudaResourceDesc resDesc = {};
//	resDesc.resType = cudaResourceTypeArray;
//	resDesc.res.array.array = arr;
//
//	if(surf != 0) CUDA_CHECK(cudaDestroySurfaceObject(surf));
//	CUDA_CHECK(cudaCreateSurfaceObject(&surf, &resDesc));
//}

//void GISparseVoxelOctree::CreateTexFromArray(cudaArray_t& arr, cudaTextureObject_t& tex)
//{
//	// Texture of SVO Dense
//	cudaResourceDesc resDesc = {};
//	cudaTextureDesc texDesc = {};
//	resDesc.resType = cudaResourceTypeArray;
//	resDesc.res.array.array = arr;
//
//	texDesc.addressMode[0] = cudaAddressModeWrap;
//	texDesc.addressMode[1] = cudaAddressModeWrap;
//	texDesc.addressMode[2] = cudaAddressModeWrap;
//	texDesc.filterMode = cudaFilterModePoint;
//	texDesc.readMode = cudaReadModeElementType;
//	texDesc.normalizedCoords = 0;
//
//	if(tex != 0) CUDA_CHECK(cudaDestroyTextureObject(tex));
//	CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
//}

//void GISparseVoxelOctree::CopyFromBufferToTex(cudaArray_t& arr, unsigned int* devPtr)
//{
//	// Copy Dense to Texture
//	cudaMemcpy3DParms params = {0};
//	params.dstArray = arr;
//	params.srcPtr =
//	{
//		devPtr,
//		octreeParams.DenseSize * sizeof(unsigned int),
//		octreeParams.DenseSize,
//		octreeParams.DenseSize
//	};
//	params.extent = {octreeParams.DenseSize, octreeParams.DenseSize, octreeParams.DenseSize};
//	params.kind = cudaMemcpyDeviceToDevice;
//	CUDA_CHECK(cudaMemcpy3D(&params));
//}
//
//void GISparseVoxelOctree::CreateTexLayeredFromArray(cudaMipmappedArray_t& arr, 
//													cudaTextureObject_t& tex)
//{
//	// Texture of SVO Dense
//	cudaResourceDesc resDesc = {};
//	cudaTextureDesc texDesc = {};
//	resDesc.resType = cudaResourceTypeMipmappedArray;
//	resDesc.res.mipmap.mipmap = arr;
//
//	texDesc.addressMode[0] = cudaAddressModeWrap;
//	texDesc.addressMode[1] = cudaAddressModeWrap;
//	texDesc.addressMode[2] = cudaAddressModeWrap;
//	texDesc.filterMode = cudaFilterModePoint;
//	texDesc.readMode = cudaReadModeElementType;
//	texDesc.normalizedCoords = 1;
//
//	if(tex != 0) CUDA_CHECK(cudaDestroyTextureObject(tex));
//	CUDA_CHECK(cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr));
//}

//void GISparseVoxelOctree::ConstructDense()
//{
//	//// Level 0 does not gurantee lower cascade parents
//	//// Each Allocator tries to allocate its parent
//	//for(unsigned int i = 0; i < allocators.size(); i++)
//	//{
//	//	uint32_t gridSize = ((allocators[i]->NumPages() * GI_PAGE_SIZE) + 
//	//							GI_THREAD_PER_BLOCK - 1) /
//	//							GI_THREAD_PER_BLOCK;
//	//	SVOReconstructDetermineNode<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//	(
//	//		dSVODense,
//	//		allocators[i]->GetVoxelPagesDevice(),
//	//	
//	//		i,
//	//		*dSVOConstants.Data()
//	//	);
//	//	CUDA_KERNEL_CHECK();
//	//}
//
//	//uint32_t gridSize = ((GI_DENSE_SIZE_CUBE) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
//	//SVOReconstructAllocateLevel<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//(
//	//	dSVODense,
//	//	*(dSVOLevelSizes.Data() + 1),
//	//	*(dSVOLevelTotalSizes.Data() + 1),
//	//	*(dSVOLevelTotalSizes.Data()),
//	//	*dSVOConstants.Data()
//	//);
//	//CUDA_KERNEL_CHECK();
//}
//
//void GISparseVoxelOctree::ConstructLevel(unsigned int currentLevel,
//										 unsigned int allocatorOffset)
//{
//	// Early Bail check 
//	unsigned int currentLevelIndex = currentLevel - octreeParams.DenseLevel;
//	CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data() + currentLevelIndex,
//						  dSVOLevelSizes.Data() + currentLevelIndex,
//						  sizeof(unsigned int),
//						  cudaMemcpyDeviceToHost));
//	if(hSVOLevelSizes[currentLevelIndex] == 0) return;
//
//	// ChildBitSet your Level (with next level's child)
//	// Allocate next level
//	// Memcopy next level start location to array
//	// Only ChildBitSet Upper Level
//	// Then Allocate your level
//	// Average Color to the level
//	//for(unsigned int i = allocatorOffset; i < allocators.size(); i++)
//	//{
//	//	//uint32_t gridSize = ((allocators[i]->NumPages() * GI_PAGE_SIZE) + 
//	//	//					 GI_THREAD_PER_BLOCK - 1) /
//	//	//					 GI_THREAD_PER_BLOCK;
//
//	//	//SVOReconstructDetermineNode<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//	//(
//	//	//	dSVOSparse,
//	//	//	tSVODenseNode,
//	//	//	allocators[i]->GetVoxelPagesDevice(),
//	//	//	dSVOOffsets,
//
//	//	//	i,
//	//	//	currentLevel,
//	//	//	*dSVOConstants.Data()
//	//	//);
//	//	//CUDA_KERNEL_CHECK();
//	//}
//	
//	//uint32_t gridSize = (hSVOLevelSizes[currentLevelIndex] + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
//	//SVOReconstructAllocateLevel<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//(
//	//	dSVOSparse + svoLevelOffsets.CPUData()[currentLevelIndex],
//	//	*(dSVOLevelSizes.Data() + currentLevelIndex + 1),
//	//	*(dSVOLevelTotalSizes.Data() + currentLevelIndex + 1),
//	//	*(dSVOLevelSizes.Data() + currentLevelIndex),
//	//	*dSVOConstants.Data()
//	//);
//	//CUDA_KERNEL_CHECK();
//}
//
//double GISparseVoxelOctree::ConstructFullAtomic(const IEVector3& ambientColor, const InjectParams& p)
//{
//	//CudaTimer timer;
//	//timer.Start();
//
//	//// Fully Atomic Version
//	//for(unsigned int i = 0; i < allocators.size(); i++)
//	//{
//	//	uint32_t nodeCount = allocators[i]->NumPages() * GI_PAGE_SIZE;
//	//	uint32_t gridSize = (nodeCount + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
//	//	SVOReconstruct<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//	(
//	//		dSVOMaterial,
// //           //dSVOLight,
//	//		dSVOSparse,
//	//		dSVODense,
//	//		dSVOLevelSizes.Data(),
//
//	//		dSVOOffsets,
//	//		dSVOLevelTotalSizes.Data(),
//	//			
//	//		// VoxelSystem Data
//	//		allocators[i]->GetVoxelPagesDevice(),
//	//		allocators[i]->GetObjRenderCacheDevice(),
//
//	//		//{ambientColor.getX(), ambientColor.getY(), ambientColor.getZ()},
//	//		0,
//	//		i,
//	//		*dSVOConstants.Data(),
//
// //           p.inject,
// //           p.span,
// //           p.outerCascadePos,
// //           float3{ambientColor.getX(), ambientColor.getY(), ambientColor.getZ()},
//
// //           p.camPos,
// //           p.camDir,
//
// //           dLightVPArray,
// //           dLightParamArray,
//
// //           p.depthNear,
// //           p.depthFar,
//
// //           tShadowMapArray,
// //           1
//	//	);
//	//	CUDA_KERNEL_CHECK();
//	//}
//	//// Copy Level Sizes
//	//CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data(),
//	//					  dSVOLevelSizes.Data(),
//	//					  hSVOLevelSizes.size() * sizeof(uint32_t),
//	//					  cudaMemcpyDeviceToHost));
//
//	//CopyFromBufferToTex(dSVODenseNodeArray, dSVODense);
//	//timer.Stop();
//	//return timer.ElapsedMilliS();
//	return 0;
//}
//
//double GISparseVoxelOctree::ConstructLevelByLevel(const IEVector3& ambientColor, const InjectParams& p)
//{
//	//CudaTimer timer;
//	//timer.Start();
//
//	//// Start with constructing dense
//	//ConstructDense();
//	//CopyFromBufferToTex(dSVODenseNodeArray, dSVODense);
//
//	//// Construct Levels
//	//for(unsigned int i = GI_DENSE_LEVEL + 1; i < allocatorGrids[0]->depth; i++)
//	//{
//	//	ConstructLevel(i, 0);
//	//}
//
//	//// Now adding cascade levels
//	//for(unsigned int i = 1; i < allocators.size(); i++)
//	//{
//	//	unsigned int currentLevel = allocatorGrids[0]->depth + i - 1;
//	//	ConstructLevel(currentLevel, i);
//	//}
//
//	//// Memcpy Last Total Size
//	//CUDA_CHECK(cudaMemcpy(hSVOLevelSizes.data() + (hSVOConstants.totalDepth - GI_DENSE_LEVEL),
//	//					  dSVOLevelSizes.Data() + (hSVOConstants.totalDepth - GI_DENSE_LEVEL),
//	//					  sizeof(uint32_t),
//	//					  cudaMemcpyDeviceToHost));
//
//	//// Average Leafs
//	//for(unsigned int i = 0; i < allocators.size(); i++)
//	//{
//	//	assert(allocators[i]->IsGLMapped() == true);
//	//	uint32_t gridSize = (allocators[i]->NumPages() * GI_PAGE_SIZE +  GI_THREAD_PER_BLOCK - 1) / 
//	//						GI_THREAD_PER_BLOCK;
//	//			
//	//	// Average Leaf Node
//	//	SVOReconstructMaterialLeaf<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	//	(
//	//		dSVOMaterial,
//
//	//		// Const SVO Data
//	//		dSVOSparse,
//	//		dSVOOffsets,
//	//		tSVODenseNode,
//
//	//		// Page Data
//	//		allocators[i]->GetVoxelPagesDevice(),
//	//									  
//	//		// For Color Lookup
//	//		allocators[i]->GetObjRenderCacheDevice(),
//
//	//		// Constants
//	//		0,
//	//		i,
//	//		*dSVOConstants.Data(),
//
// //           p.inject,
// //           p.span,
// //           p.outerCascadePos,
// //           float3{ambientColor.getX(), ambientColor.getY(), ambientColor.getZ()},
//
// //           p.camPos,
// //           p.camDir,
//
// //           dLightVPArray,
// //           dLightParamArray,
//
// //           p.depthNear,
// //           p.depthFar,
//
// //           tShadowMapArray,
// //           1
//	//	);
//	//	CUDA_KERNEL_CHECK();
//	//}
//
//	//timer.Stop();
//	//return timer.ElapsedMilliS();
//	return 0.0;
//}
//
//double GISparseVoxelOctree::LightInject(InjectParams params,
//										const std::vector<IEMatrix4x4>& projMatrices,
//										const std::vector<IEMatrix4x4>& invViewProj)
//{
//    return 0.0;
//}
//
//double GISparseVoxelOctree::AverageNodes()
//{
//	////CudaTimer timer;
//	////timer.Start();
//
//	////// Now use leaf nodes to average upper nodes
//	////// Start bottom up
//	////for(int i = hSVOConstants.totalDepth - 1; i >= static_cast<int>(hSVOConstants.denseDepth); i--)
//	////{
//	////	unsigned int arrayIndex = i - GI_DENSE_LEVEL;
//	////	unsigned int levelDim = GI_DENSE_SIZE >> (GI_DENSE_LEVEL - i);
//	////	unsigned int levelSize = (i > GI_DENSE_LEVEL) ? hSVOLevelSizes[arrayIndex]: 
//	////													levelDim * levelDim * levelDim;
//	////	if(levelSize == 0) continue;
//
//	////	uint32_t gridSize = (levelSize * 2 + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
//	////	// Average Level
//	////	SVOReconstructAverageNode<<<gridSize, GI_THREAD_PER_BLOCK>>>
//	////	(
//	////		dSVOMaterial,
//	////		sSVODenseMat[0],
//
// ////           //dSVOLight,
//	////		dSVODense,
//	////		dSVOSparse,
//
//	////		dSVOOffsets,
//	////		*(dSVOOffsets + arrayIndex),
//	////		*(dSVOOffsets + arrayIndex + 1),
//
//	////		levelSize,
//	////		0,
//	////		i,
//	////		*dSVOConstants.Data()
//	////	);
//	////	CUDA_KERNEL_CHECK();
//	////}
//	////
//	////// Dense Reduction
//	////for(int i = 1; i < GI_DENSE_TEX_COUNT; i++)
//	////{
//	////	uint32_t levelSize = GI_DENSE_SIZE >> i;
//	////	uint32_t levelSizeCube = levelSize * levelSize * levelSize;
//	////		
//	////	uint32_t grid = ((levelSizeCube * GI_DENSE_WORKER_PER_PARENT) + GI_THREAD_PER_BLOCK - 1) / 
//	////					GI_THREAD_PER_BLOCK;
//	////	
//	////	SVOReconstructAverageNode<<<grid, GI_THREAD_PER_BLOCK>>>
//	////	(
//	////		sSVODenseMat[i - 1],
//	////		sSVODenseMat[i],
//	////		levelSize
//	////	);
//	////}
//
//	////timer.Stop();
//	////return timer.ElapsedMilliS();
//	//return 0.0;
//	return 0;
//}
//
//void GISparseVoxelOctree::UpdateSVO(double& reconstTime,
//									double& injectTime,
//									double& averageTime,
//									const IEVector3& ambientColor,
//									const InjectParams& p,
//									const std::vector<IEMatrix4x4>& projMatrices,
//									const std::vector<IEMatrix4x4>& invViewProj)
//{
//	//// Clear Mat Texture
//	//GLuint ff[4] = {0x0, 0x0, 0x0, 0x0};
//	//for(unsigned int i = 0; i < DenseLevelCount; i++)
//	//	glClearTexImage(svoDenseMat, i, GL_RGBA_INTEGER, GL_UNSIGNED_INT, &ff);
//
//	//// Shadow Maps
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &sceneLightParamResource));
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &sceneVPMatrixResource));
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &sceneShadowMapResource));
//
//	//// SVO Nodes
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &svoMaterialResource));
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &svoNodeResource));
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &svoLevelOffsetResource));
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &svoDenseNodeResource));
//	//CUDA_CHECK(cudaGraphicsMapResources(1, &svoDenseTexResource));
//	//
//	//size_t size;
//	//CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVODense), 
//	//												 &size, svoNodeResource));
//	//CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVOMaterial),
//	//												 &size, svoMaterialResource));
//	//CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dSVOOffsets),
//	//												&size, svoLevelOffsetResource));
//	//CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&dSVODenseNodeArray, svoDenseNodeResource, 0, 0));
//	//CreateSurfFromArray(dSVODenseNodeArray, sSVODenseNode);
//	//CreateTexFromArray(dSVODenseNodeArray, tSVODenseNode);
//	//for(unsigned int i = 0; i < DenseLevelCount; i++)
//	//{
//	//	CUDA_CHECK(cudaGraphicsSubResourceGetMappedArray(&dSVODenseMatArray[i], svoDenseTexResource, 0, i));
//	//	CreateSurfFromArray(dSVODenseMatArray[i], sSVODenseMat[i]);
//	//}
//	//dSVOSparse = dSVODense + DenseSizeCube;
//
//	//// Shadow Related
//	//CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dLightParamArray),
//	//												&size, sceneLightParamResource));
//	//CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dLightVPArray),
//	//												&size, sceneVPMatrixResource));
//	//CUDA_CHECK(cudaGraphicsResourceGetMappedMipmappedArray(&shadowMapArray, sceneShadowMapResource));
//	//CreateTexLayeredFromArray(shadowMapArray, tShadowMapArray);
//
//	//// Reset Atomic Counter since we reconstruct every frame
//	//uint32_t usedNodeCount = hSVOLevelSizes.back() + svoLevelOffsets.CPUData().back();
//	//CUDA_CHECK(cudaMemset(dSVODense, 0xFF, sizeof(CSVONode) * (usedNodeCount + GI_DENSE_SIZE_CUBE)));
//	//CUDA_CHECK(cudaMemset(dSVOMaterial, 0x00, sizeof(CSVOMaterial) * (usedNodeCount)));
//
//	//dSVOLevelSizes.Memset(0x00, 0, dSVOLevelSizes.Size());
//	//std::fill(hSVOLevelSizes.begin(), hSVOLevelSizes.end(), 0);
//
//	//// Maxwell is faster with fully atomic code (CAS Locks etc.)
//	//// However kepler sucks(660ti) (100ms compared to 5ms) 
// //   IEVector3 aColor = (false) ? IEVector3::ZeroVector : ambientColor;
//	//if(CudaInit::CapabilityMajor() >= 5)
//	//	reconstTime = ConstructFullAtomic(aColor, p);
//	//else
//	//	reconstTime = ConstructLevelByLevel(aColor, p);
// //   injectTime = 0.0;// LightInject(p, projMatrices, invViewProj);
//	//averageTime = AverageNodes();
//
//	////// DEBUG
//	////GI_LOG("-------------------------------------------");
//	////GI_LOG("Tree Node Data");
//	////unsigned int i;
//	////for(i = 0; i <= allocatorGrids[0]->depth - GI_DENSE_LEVEL + allocators.size() - 1; i++)
//	////{
//	////	if(i == 0) GI_LOG("#%d Dense : %d", GI_DENSE_LEVEL + i, GI_DENSE_SIZE_CUBE);
//	////	else GI_LOG("#%d Level : %d", GI_DENSE_LEVEL + i, hSVOLevelSizes[i]);
//	////}
//	////unsigned int total = std::accumulate(hSVOLevelSizes.begin(),
//	////									 hSVOLevelSizes.end(), 0);
//	////GI_LOG("Total : %d", total);
//	//
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &sceneLightParamResource));
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &sceneShadowMapResource));
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &sceneVPMatrixResource));
//
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoMaterialResource));
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoNodeResource));
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoLevelOffsetResource));
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoDenseNodeResource));
//	//CUDA_CHECK(cudaGraphicsUnmapResources(1, &svoDenseTexResource));
//}
//
//double GISparseVoxelOctree::GlobalIllumination(DeferredRenderer& dRenderer,
//											   const Camera& camera,
//											   SceneI& scene,
//											   float coneAngle,
//											   float maxDistance,
//											   float falloffFactor,
//											   float sampleDistanceRatio,
//											   float intensityFactorAO,
//											   float intensityFactorGI,
//											   bool giOn,
//											   bool aoOn,
//											   bool specular)
//{
////	// Light Intensity Texture
////	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
////	glClearTexImage(traceTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);
////
////	// Update FrameTransform Matrices 
////	// And its inverse realted buffer
////	//assert(TraceWidth == DeferredRenderer::gBuffWidth);
////	//assert(TraceHeight == DeferredRenderer::gBuffHeight);
////	dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
////	dRenderer.GetFTransform().Update(camera.generateTransform());
////
////	// Timing Voxelization Process
////	GLuint queryID;
////	glGenQueries(1, &queryID);
////	glBeginQuery(GL_TIME_ELAPSED, queryID);
////
////	// Set Cascade Trace Data
////	float3 pos = allocatorGrids[0]->position;
////	uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
////	uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
////	svoTraceData.CPUData()[0] =
////	{
////		{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
////		{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
////		{
////			static_cast<unsigned int>(allocators.size()),
////			GI_DENSE_SIZE_CUBE,
////			0,
////			0
////		}
////	};
////	svoTraceData.SendData();
////
////    //TEST
////    //// Convert Diameter to interpolation weight and levels
////    //float diameter = std::tan(coneAngle) * maxDistance;
////    //float diameterRatio = diameter / allocatorGrids.back()->span;
////    //diameterRatio = std::max(diameterRatio, 1.0f);
////    //unsigned int closestPow = static_cast<unsigned int>(std::floor(std::log2(diameterRatio)));
////    //float interp = (diameterRatio - float(0x1 << closestPow)) / float(0x1 << closestPow);
////    //unsigned int nodeLevel = depth - closestPow;
////    ////nodeDepth = 8;
////
////    //GI_LOG("(D%f, C%d)(%f, %d, %d)", diameterRatio, closestPow, interp, nodeLevel, nodeLevel - 1);
////
////	// Set Cone Trace Data
////	svoConeParams.CPUData()[0] =
////	{
////		{maxDistance, std::tan(coneAngle), std::tan(coneAngle * 0.5f), sampleDistanceRatio},
////		{intensityFactorAO, intensityFactorGI, IEMath::Sqrt3, falloffFactor}
////	};
////	svoConeParams.SendData();
////
////	// Shaders
////	computeGI.Bind();
////
////	// Shadow Related
////	dRenderer.BindShadowMaps(scene);
////	dRenderer.BindLightBuffers(scene);
////
////	// Uniforms
////	glUniform1ui(U_LIGHT_INDEX, static_cast<GLuint>(0));
////	glUniform1ui(U_ON_OFF_SWITCH, specular ? 1u : 0u);
////
////	// Buffers
////	svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
////	svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
////	svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
////	dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
////	dRenderer.GetFTransform().Bind();
////	svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);
////	svoConeParams.BindAsUniformBuffer(U_CONE_PARAMS);
////
////	// Images
////	dRenderer.GetGBuffer().BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
////	dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
////	dRenderer.GetGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
////	glBindImageTexture(I_LIGHT_INENSITY, traceTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8/*GL_RGBA16*/);
////	glActiveTexture(GL_TEXTURE0 + T_DENSE_NODE);
////	glBindTexture(GL_TEXTURE_3D, svoDenseNode);
////	glBindSampler(T_DENSE_NODE, nodeSampler);
////	glActiveTexture(GL_TEXTURE0 + T_DENSE_MAT);
////	glBindTexture(GL_TEXTURE_3D, svoDenseMat);
////	glBindSampler(T_DENSE_MAT, materialSampler);
////
////	// Dispatch
////	uint2 gridSize;
////    gridSize.x = (TraceWidth + 16 - 1) / 16;
////    gridSize.y = (TraceHeight + 16 - 1) / 16;
////	glDispatchCompute(gridSize.x, gridSize.y, 1);
////	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////    // Detect Edge
////    computeEdge.Bind();
////    glUniform2f(U_TRESHOLD, 0.007f, IEMath::CosF(IEMath::ToRadians(10.0f)));
////    glUniform2f(U_NEAR_FAR, camera.near, camera.far);
////    dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
////    dRenderer.GetGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
////    glBindImageTexture(I_OUT, edgeTex, 0, false, 0, GL_WRITE_ONLY, GL_RG8);
////    //glBindImageTexture(I_OUT, traceTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA16F);
////
////    gridSize.x = (TraceWidth + 16 - 1) / 16;
////    gridSize.y = (TraceHeight + 16 - 1) / 16;
////    glDispatchCompute(gridSize.x, gridSize.y, 1);
////    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////    // Edge Aware Gauss
////    computeGauss32.Bind();
////    glActiveTexture(GL_TEXTURE0 + T_EDGE);
////    glBindTexture(GL_TEXTURE_2D, edgeTex);
////    glBindSampler(T_EDGE, nodeSampler);
////
////    // Call #1 (Vertical)
////    GLuint inTex = traceTexture;
////    GLuint outTex = gaussTex;
////    for(unsigned int i = 0; i < 4; i++)
////    {
////        glActiveTexture(GL_TEXTURE0 + T_IN);
////        glBindTexture(GL_TEXTURE_2D, inTex);
////        glBindSampler(T_IN, gaussSampler);
////        glBindImageTexture(I_OUT, outTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8/*GL_RGBA16*/);
////        glUniform1ui(U_DIRECTION, 0);
////        glDispatchCompute(gridSize.x, gridSize.y, 1);
////        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////        // Call #2 (Horizontal)
////        glActiveTexture(GL_TEXTURE0 + T_IN);
////        glBindTexture(GL_TEXTURE_2D, outTex);
////        glBindSampler(T_IN, gaussSampler);
////        glBindImageTexture(I_OUT, inTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8/*GL_RGBA16*/);
////        glUniform1ui(U_DIRECTION, 1);
////        glDispatchCompute(gridSize.x, gridSize.y, 1);
////
////    }
////    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////	// Apply to DRenderer Li Tex
////	computeLIApply.Bind();
////	
////	// Uniform
////	glUniform2ui(U_ON_OFF_SWITCH, aoOn ? 1u : 0u, giOn ? 1u : 0u);
////
////	// Textures
////	GLuint gBufferLITex = dRenderer.GetLightIntensityBufferGL();
////	glBindImageTexture(I_LIGHT_INENSITY, gBufferLITex, 0, false, 0, GL_READ_WRITE, GL_RGBA16F);
////	glActiveTexture(GL_TEXTURE0 + T_COLOR);
////	glBindTexture(GL_TEXTURE_2D, traceTexture);
////	glBindSampler(T_COLOR, nodeSampler);
////
////	gridSize.x = (DeferredRenderer::gBuffWidth + 16 - 1) / 16;
////	gridSize.y = (DeferredRenderer::gBuffHeight + 16 - 1) / 16;
////	glDispatchCompute(gridSize.x, gridSize.y, 1);
////	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////	// Timer
////	GLuint64 timeElapsed = 0;
////	glEndQuery(GL_TIME_ELAPSED);
////	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
////
////	// I have to unbind the compute shader or weird things happen
////	Shader::Unbind(ShaderType::COMPUTE);
////	return timeElapsed / 1000000.0;
////}
////
////double GISparseVoxelOctree::AmbientOcclusion(DeferredRenderer& dRenderer,
////											 const Camera& camera,
////											 float coneAngle,
////											 float maxDistance,
////											 float falloffFactor,
////											 float sampleDistanceRatio,
////											 float intensityFactor)
////{
////	// Light Intensity Texture
////	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
////	glClearTexImage(traceTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);
////
////	// Update FrameTransform Matrices 
////	// And its inverse realted buffer
////	//assert(TraceWidth == DeferredRenderer::gBuffWidth);
////	//assert(TraceHeight == DeferredRenderer::gBuffHeight);
////	dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
////	dRenderer.GetFTransform().Update(camera.generateTransform());
////
////	// Timing Voxelization Process
////	GLuint queryID;
////	glGenQueries(1, &queryID);
////	glBeginQuery(GL_TIME_ELAPSED, queryID);
////
////	// Set Cascade Trace Data
////	float3 pos = allocatorGrids[0]->position;
////	uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
////	uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
////	svoTraceData.CPUData()[0] =
////	{
////		{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
////		{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
////		{
////			static_cast<unsigned int>(allocators.size()),
////			GI_DENSE_SIZE_CUBE,
////			0,
////			0
////		}
////	};
////	svoTraceData.SendData();
////
////	// Set Cone Trace Data
////	svoConeParams.CPUData()[0] =
////	{
////		{maxDistance, std::tan(coneAngle), std::tan(coneAngle * 0.5f), sampleDistanceRatio},
////		{intensityFactor, IEMath::Sqrt2, IEMath::Sqrt3, falloffFactor}
////	};
////	svoConeParams.SendData();
////
////	// Shaders
////	computeAO.Bind();
////	//computeAOSurf.Bind();
////
////	// Buffers
////	svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
////	svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
////	svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
////	dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
////	dRenderer.GetFTransform().Bind();
////	svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);
////	svoConeParams.BindAsUniformBuffer(U_CONE_PARAMS);
////
////	// Images
////	dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
////	dRenderer.GetGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
////	glBindImageTexture(I_LIGHT_INENSITY, traceTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
////	glActiveTexture(GL_TEXTURE0 + T_DENSE_NODE);
////	glBindTexture(GL_TEXTURE_3D, svoDenseNode);
////	glBindSampler(T_DENSE_NODE, nodeSampler);
////	glActiveTexture(GL_TEXTURE0 + T_DENSE_MAT);
////	glBindTexture(GL_TEXTURE_3D, svoDenseMat);
////	glBindSampler(T_DENSE_MAT, materialSampler);
////	
////	// Dispatch
////	uint2 gridSize;
////	gridSize.x = (TraceWidth + 16 - 1) / 16;
////	gridSize.y = (TraceHeight + 16 - 1) / 16;
////	glDispatchCompute(gridSize.x, gridSize.y, 1);
////
////	//uint2 gridSize;
////	//gridSize.x = (TraceWidth + 16 - 1) / 16;
////	//gridSize.y = (TraceHeight + 16 - 1) / 16;
////	//glDispatchCompute(gridSize.x, gridSize.y, 1);
////
////	//// Detect Edge
////	//computeEdge.Bind();
////	//glUniform2f(U_TRESHOLD, 0.007f, IEMath::CosF(IEMath::ToRadians(20.0f)));
////	//glUniform2f(U_NEAR_FAR, camera.near, camera.far);
////	//dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
////	//dRenderer.GetGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
////	//glBindImageTexture(I_OUT, edgeTex, 0, false, 0, GL_WRITE_ONLY, GL_RG8);
////	//
////	//gridSize.x = (TraceWidth + 16 - 1) / 16;
////	//gridSize.y = (TraceHeight + 16 - 1) / 16;
////	//glDispatchCompute(gridSize.x, gridSize.y, 1);
////	//glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////	////dRenderer.ShowTexture(camera, edgeTex);
////
////	//// Edge Aware Gauss
////	//computeGauss32.Bind();
////	//glActiveTexture(GL_TEXTURE0 + T_EDGE);
////	//glBindTexture(GL_TEXTURE_2D, svoDenseMat);
////	//glBindSampler(T_EDGE, gaussSampler);
////
////	//// Call #1 (Vertical)
////	//GLuint inTex = liTexture;
////	//GLuint outTex = gaussTex;
////	//for(unsigned int i = 0; i < 32; i++)
////	//{
////	//	glActiveTexture(GL_TEXTURE0 + T_IN);
////	//	glBindTexture(GL_TEXTURE_2D, inTex);
////	//	glBindSampler(T_EDGE, gaussSampler);
////	//	glBindImageTexture(I_OUT, outTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
////	//	glUniform1ui(U_DIRECTION, 0);
////	//	glDispatchCompute(gridSize.x, gridSize.y, 1);
////	//	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////
////	//	// Call #2 (Horizontal)
////	//	glActiveTexture(GL_TEXTURE0 + T_IN);
////	//	glBindTexture(GL_TEXTURE_2D, outTex);
////	//	glBindSampler(T_EDGE, gaussSampler);
////	//	glBindImageTexture(I_OUT, inTex, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
////	//	glUniform1ui(U_DIRECTION, 1);
////	//	glDispatchCompute(gridSize.x, gridSize.y, 1);
////
////	//	GLuint temp = inTex;
////	//	inTex = outTex;
////	//	outTex = temp;
////	//}
////
////	// Render to window
////	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
////	dRenderer.ShowTexture(camera, traceTexture);
////
////	// Timer
////	GLuint64 timeElapsed = 0;
////	glEndQuery(GL_TIME_ELAPSED);
////	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
////
////	// I have to unbind the compute shader or weird things happen
////	Shader::Unbind(ShaderType::COMPUTE);
////	return timeElapsed / 1000000.0;
//	return 0.0;
//}
//
//double GISparseVoxelOctree::DebugDeferredSVO(DeferredRenderer& dRenderer,
//											 const Camera& camera,
//											 uint32_t renderLevel,
//											 SVOTraceType type)
//{
//	//// Update FrameTransform Matrices 
//	//// And its inverse realted buffer
//	//assert(TraceWidth == DeferredRenderer::gBuffWidth);
//	//assert(TraceHeight == DeferredRenderer::gBuffHeight);
//	//dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
//	//dRenderer.GetFTransform().Update(camera.generateTransform());
//
//	//// Timing Voxelization Process
//	//GLuint queryID;
//	//glGenQueries(1, &queryID);
//	//glBeginQuery(GL_TIME_ELAPSED, queryID);
//
//	//// Set Cascade Trace Data
//	//float3 pos = allocatorGrids[0]->position;
//	//uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
//	//uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
//	//svoTraceData.CPUData()[0] =
//	//{
//	//	{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
//	//	{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
//	//	{
//	//		static_cast<unsigned int>(allocators.size()),
//	//		GI_DENSE_SIZE * GI_DENSE_SIZE * GI_DENSE_SIZE,
//	//		0,
//	//		GI_DENSE_LEVEL - GI_DENSE_TEX_COUNT + 1
//	//	}
//	//};
//	//svoTraceData.SendData();
//
//	//// Shaders
//	////computeVoxTraceDeferred.Bind();
//	//computeVoxTraceDeferredLerp.Bind();
//	//glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(type));
//	//glUniform1ui(U_FETCH_LEVEL, static_cast<GLuint>(renderLevel));
//
//	//// Buffers
//	//svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
//	//svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
//	//svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
//	//dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
//	//dRenderer.GetFTransform().Bind();
//	//svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);
//
//	//// Images
//	//dRenderer.GetGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
//	//glBindImageTexture(I_COLOR_FB, traceTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
//	//glActiveTexture(GL_TEXTURE0 + T_DENSE_NODE);
//	//glBindTexture(GL_TEXTURE_3D, svoDenseNode);
//	//glBindSampler(T_DENSE_NODE, nodeSampler);
//	//glActiveTexture(GL_TEXTURE0 + T_DENSE_MAT);
//	//glBindTexture(GL_TEXTURE_3D, svoDenseMat);
//	//glBindSampler(T_DENSE_MAT, materialSampler);
//
//	//// Dispatch
//	//uint2 gridSize;
//	//gridSize.x = (TraceWidth + 16 - 1) / 16;
//	//gridSize.y = (TraceHeight + 16 - 1) / 16;
//	//glDispatchCompute(gridSize.x, gridSize.y, 1);
//
//	//// Render to window
//	//glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
//	//dRenderer.ShowTexture(camera, traceTexture);
//
//	//// Timer
//	//GLuint64 timeElapsed = 0;
//	//glEndQuery(GL_TIME_ELAPSED);
//	//glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
//
//	//// I have to unbind the compute shader or weird things happen
//	//Shader::Unbind(ShaderType::COMPUTE);
//	//return static_cast<double>(timeElapsed) / 1000000.0;
//	return 0.0;
//}
//
//double GISparseVoxelOctree::DebugTraceSVO(DeferredRenderer& dRenderer,
//										  const Camera& camera,
//										  uint32_t renderLevel,
//										  SVOTraceType type)
//{
//	//// Update FrameTransform Matrices 
//	//// And its inverse realted buffer
//	//dRenderer.RefreshInvFTransform(camera, TraceWidth, TraceHeight);
//	//dRenderer.GetFTransform().Update(camera.generateTransform());
//
//	//// Timing Voxelization Process
//	//GLuint queryID;
//	//glGenQueries(1, &queryID);
//	//glBeginQuery(GL_TIME_ELAPSED, queryID);
//	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
//
//	//// Set Cascade Trace Data
//	//float3 pos = allocatorGrids[0]->position;
//	//uint32_t dim = allocatorGrids[0]->dimension.x * (0x1 << (allocators.size() - 1));
//	//uint32_t depth = allocatorGrids[0]->depth + static_cast<uint32_t>(allocators.size()) - 1;
//	//svoTraceData.CPUData()[0] = 
//	//{
//	//	{pos.x, pos.y, pos.z, allocatorGrids.back()->span},
//	//	{dim, depth, GI_DENSE_SIZE, GI_DENSE_LEVEL},
//	//	{
//	//		static_cast<unsigned int>(allocators.size()), 
//	//		GI_DENSE_SIZE_CUBE,
//	//		0,
//	//		GI_DENSE_LEVEL - GI_DENSE_TEX_COUNT + 1
//	//	}
//	//};
//	//svoTraceData.SendData();
//
//	//// Shaders
//	//computeVoxTraceWorld.Bind();
//	//glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(type));
//	//glUniform1ui(U_FETCH_LEVEL, static_cast<GLuint>(renderLevel));
//
//	//// Buffers
//	//svoNodeBuffer.BindAsShaderStorageBuffer(LU_SVO_NODE);
//	//svoMaterialBuffer.BindAsShaderStorageBuffer(LU_SVO_MATERIAL);
//	//svoLevelOffsets.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET);
//	//dRenderer.GetInvFTransfrom().BindAsUniformBuffer(U_INVFTRANSFORM);
//	//dRenderer.GetFTransform().Bind();
//	//svoTraceData.BindAsUniformBuffer(U_SVO_CONSTANTS);
//
//	//// Images
//	//glBindImageTexture(I_COLOR_FB, traceTexture, 0, false, 0, GL_WRITE_ONLY, GL_RGBA8);
//	//glActiveTexture(GL_TEXTURE0 + T_DENSE_NODE);
//	//glBindTexture(GL_TEXTURE_3D, svoDenseNode);
//	//glBindSampler(T_DENSE_NODE, nodeSampler);
//	//glActiveTexture(GL_TEXTURE0 + T_DENSE_MAT);
//	//glBindTexture(GL_TEXTURE_3D, svoDenseMat);
//	//glBindSampler(T_DENSE_MAT, materialSampler);
//
//	//// Dispatch
//	//uint2 gridSize;
//	//gridSize.x = (TraceWidth + 16 - 1) / 16;
//	//gridSize.y = (TraceHeight + 16 - 1) / 16;
//	//glDispatchCompute(gridSize.x, gridSize.y, 1);
//	//
//	//// Render to window
//	//glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
//	//dRenderer.ShowTexture(camera, traceTexture);
//
//	//// Timer
//	//GLuint64 timeElapsed = 0;
//	//glEndQuery(GL_TIME_ELAPSED);
//	//glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
//	//
//	//// I have to unbind the compute shader or weird things happen
//	//Shader::Unbind(ShaderType::COMPUTE);
//	//return timeElapsed / 1000000.0;
//	return 0.0;
//}
//
//uint64_t GISparseVoxelOctree::MemoryUsage() const
//{
//	//uint64_t totalBytes = 0;
//	//totalBytes += svoNodeBuffer.Capacity() * sizeof(CSVONode);
//	//totalBytes += svoMaterialBuffer.Capacity() * sizeof(CSVOMaterial);
//	//totalBytes += dSVOLevelSizes.Size() * sizeof(unsigned int);
//	//totalBytes += sizeof(unsigned int);
//	//totalBytes += GI_DENSE_SIZE_CUBE * sizeof(CSVONode);	// Dense Tex
//	//for(unsigned int i = 0; i < GI_DENSE_TEX_COUNT; i++)
//	//{
//	//	size_t texSize = GI_DENSE_SIZE >> i;
//	//	totalBytes += sizeof(CSVOMaterial) * texSize * texSize * texSize;
//	//}
//	//return totalBytes;
//	return 0;
//}
//
//uint32_t GISparseVoxelOctree::MinLevel() const
//{
//	//return hSVOConstants.denseDepth - GI_DENSE_TEX_COUNT + 1;
//	return 0;
//}
//
//uint32_t GISparseVoxelOctree::MaxLevel() const
//{
//	return hSVOConstants.totalDepth;
//}
//
//const CSVOConstants& GISparseVoxelOctree::SVOConsts() const
//{
//	return hSVOConstants;
//}
