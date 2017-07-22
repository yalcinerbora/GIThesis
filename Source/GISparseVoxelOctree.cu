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
#include "ConeTraceTexture.h"
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
	//CudaTimer t;
	//t.Start();

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
	texDesc.normalizedCoords = 1;

	CUDA_CHECK(cudaCreateTextureObject(&tShadowMapArray, &resDesc, &texDesc, nullptr));

	// Buffer
	size_t size;
	uint8_t* glBufferCUDA = nullptr;
	CUDA_CHECK(cudaGraphicsMapResources(1, &lightBufferResource));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glBufferCUDA),
													&size, lightBufferResource));

	dLightParamArray = reinterpret_cast<const CLight*>(glBufferCUDA + lightOffset);
	dLightVPMatrixArray = reinterpret_cast<const CMatrix4x4*>(glBufferCUDA + matrixOffset);

	//t.Stop();
	//GI_LOG("Mapping Shadow Maps %f ms", t.ElapsedMilliS());
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
	, hLevelSizes(octreeParams.MaxSVOLevel + 1, 0)
	, nodeIllumDifference(0)
	, shadowMaps(currentScene->getSceneLights())
	, compVoxTraceWorld(ShaderType::COMPUTE, "Shaders/VoxTraceWorld.comp")
	, compVoxSampleWorld(ShaderType::COMPUTE, "Shaders/VoxTraceDeferred.comp")
	, compGI(ShaderType::COMPUTE, "Shaders/VoxGI.comp")
{	
	// Generate Initial Sizes for each level
	std::vector<uint32_t> levelCapacities(octreeParams.MaxSVOLevel + 1, 0);
	std::vector<uint32_t> internalOffsets(octreeParams.MaxSVOLevel + 1, 0);
	//std::vector<uint32_t> voxPackedOffsets(octreeParams.MaxSVOLevel + 1, 0);
	size_t offset = 0;
	for(uint32_t i = octreeParams.MinSVOLevel; i <= octreeParams.MaxSVOLevel; i++)
	{
		assert(sizes[i] % 8 == 0);
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
	hIllumOffsetsAndCapacities.insert(hIllumOffsetsAndCapacities.end(),
									  internalOffsets.begin(), internalOffsets.end());
	hIllumOffsetsAndCapacities.insert(hIllumOffsetsAndCapacities.end(),
									  levelCapacities.begin(), levelCapacities.end());
		
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
	offset += (totalNodeSize - levelCapacities[octreeParams.MaxSVOLevel]) * sizeof(CSVONode);
	// Illum Data
	offset = DeviceOGLParameters::SSBOAlignOffset(offset);
	illumOffset = offset;
	offset += (totalIllumSize) * sizeof(CSVOIllumination);
	
	// Offsets Generated Allocate
	oglData.Resize(offset, false);

	// Send Offset Pointers
	oglData.SendSubData(reinterpret_cast<const uint8_t*>(internalOffsets.data()),
						static_cast<uint32_t>(illumOffsetsOffset), 
						(octreeParams.MaxSVOLevel + 1) * sizeof(uint32_t));

	// Now CUDA
	size_t totalSize = (octreeParams.MaxSVOLevel + 1) * (sizeof(uint32_t) * 2 +
														 sizeof(CSVOLevel));
	totalSize += (totalNodeSize - levelCapacities[octreeParams.MaxSVOLevel]) * sizeof(uint32_t);
	cudaData.Resize(totalSize);

	// Allocation of CUDA Side
	size_t voxPackStart;
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
	// Vox Packed
	voxPackStart = offset;
	offset += (totalNodeSize - levelCapacities[octreeParams.MaxSVOLevel]) * sizeof(uint32_t);
	assert(offset == totalSize);

	// Send pointers of packed data
	std::vector<uint32_t*> voxPointers(octreeParams.MaxSVOLevel + 1, nullptr);
	for(uint32_t i = octreeParams.DenseLevel; i < octreeParams.MaxSVOLevel; i++)
	{
		uint32_t offset = static_cast<uint32_t>(hIllumOffsetsAndCapacities[i] - nodeIllumDifference);
		voxPointers[i] = reinterpret_cast<uint32_t*>(cudaData.Data() + voxPackStart) + offset;
	}
	CUDA_CHECK(cudaMemset(cudaData.Data() + voxPackStart, 0xFF,
				          (totalNodeSize - levelCapacities[octreeParams.MaxSVOLevel]) * sizeof(uint32_t)));
	// Lets be sure here about data offsets etc...
	uint8_t* dOctreeLevelsOffset = reinterpret_cast<uint8_t*>(dOctreeLevels) + offsetof(CSVOLevel, gVoxId);
	CUDA_CHECK(cudaMemcpy2D(dOctreeLevelsOffset, sizeof(CSVOLevel),
							voxPointers.data(), sizeof(uint32_t*),
							sizeof(uint32_t*), octreeParams.MaxSVOLevel + 1,
							cudaMemcpyHostToDevice));

	// Load Level Capacities
	CUDA_CHECK(cudaMemcpy(const_cast<uint32_t*>(dLevelCapacities),
						  levelCapacities.data(),
						  (octreeParams.MaxSVOLevel + 1) * sizeof(uint32_t),
						  cudaMemcpyHostToDevice));

	// Register CUDA Resource
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&gpuResource, oglData.getGLBuffer(),
											cudaGraphicsMapFlagsNone));


	// Memset
	CUDA_CHECK(cudaGraphicsMapResources(1, &gpuResource));
	size_t size; uint8_t* oglCudaPtr;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&oglCudaPtr),
													&size, gpuResource));
	assert(size == oglData.Capacity());
	CUDA_CHECK(cudaMemset(oglCudaPtr + illumOffset, 0x00, oglData.Capacity() - illumOffset));
	CUDA_CHECK(cudaMemset(oglCudaPtr + nodeOffset, 0xFF, illumOffset - nodeOffset));

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &gpuResource));
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
	, hLevelSizes(std::move(other.hLevelSizes))
	, hIllumOffsetsAndCapacities(std::move(other.hIllumOffsetsAndCapacities))
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
	hLevelSizes = std::move(other.hLevelSizes);
	hIllumOffsetsAndCapacities = std::move(other.hIllumOffsetsAndCapacities);
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
	//CudaTimer t;
	//t.Start();

	// Get Node Pointer
	CUDA_CHECK(cudaGraphicsMapResources(1, &gpuResource));
	size_t size; uint8_t* oglCudaPtr;
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&oglCudaPtr),
													&size, gpuResource));
	//assert(size == oglData.Capacity());
	std::vector<CSVOLevel> svoLevels(octreeParams->MaxSVOLevel + 1, {nullptr, nullptr, nullptr});
	for(uint32_t i = octreeParams->MinSVOLevel; i < octreeParams->MaxSVOLevel + 1; i++)
	{		
		CSVONode* nodePtr = nullptr;
		CSVOIllumination* illumPtr = reinterpret_cast<CSVOIllumination*>(oglCudaPtr + illumOffset)
																		 + hIllumOffsetsAndCapacities[i];

		uint32_t denseSize = hIllumOffsetsAndCapacities[(octreeParams->MaxSVOLevel + 1) + i];
		uint32_t size = (i <= octreeParams->DenseLevel) ? denseSize : hLevelSizes[i];
		if(i >= octreeParams->DenseLevel && i < octreeParams->MaxSVOLevel)
		{
			size_t nodeLevelOffset = (hIllumOffsetsAndCapacities[i] - nodeIllumDifference);
			nodePtr = reinterpret_cast<CSVONode*>(oglCudaPtr + nodeOffset) + nodeLevelOffset;

			// Clear used node pointers			
			CUDA_CHECK(cudaMemset(nodePtr, 0xFF, size * sizeof(CSVONode)));
			CUDA_CHECK(cudaMemset(nodePtr, 0xFF, size * sizeof(CSVONode)));
		}
		svoLevels[i].gLevelNodes = nodePtr;
		svoLevels[i].gLevelIllum = illumPtr;

		// Clear used illum
		CUDA_CHECK(cudaMemset(illumPtr, 0x00, size * sizeof(CSVOIllumination)));
	}

	// Print Allocator Usage
	//PrintSVOLevelUsages(hLevelSizes);

	// Clear level allocators
	CUDA_CHECK(cudaMemset(dLevelSizes, 0x00, (octreeParams->MaxSVOLevel + 1) * sizeof(uint32_t)));

	// Copy Generated Pointers
	CUDA_CHECK(cudaMemcpy2D(dOctreeLevels, sizeof(CSVOLevel),
							svoLevels.data(), sizeof(CSVOLevel),
							sizeof(CSVONode*) + sizeof(CSVOIllumination*), 
							octreeParams->MaxSVOLevel + 1,
							cudaMemcpyHostToDevice));

	//t.Stop();
	//GI_LOG("Map Time (with clear) %f ms", t.ElapsedMilliS());
}

void GISparseVoxelOctree::UnmapOGLData()
{
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &gpuResource));
}

void GISparseVoxelOctree::PrintSVOLevelUsages(const std::vector<uint32_t>& svoSizes) const
{
	for(size_t i = octreeParams->DenseLevel + 1; i < svoSizes.size(); i++)
	{
		uint32_t size = svoSizes[i];
		uint32_t capacity = hIllumOffsetsAndCapacities[i + (octreeParams->MaxSVOLevel + 1)];

		printf("Level #%2d    %9d / %9d", static_cast<int>(i), size, capacity);
		if(size >= capacity) GI_LOG(" OVERFLOW!");
		else GI_LOG("");
	}
	GI_LOG("----------");
}

double GISparseVoxelOctree::GenerateHierarchy(bool doTiming,
											  // Page System
											  const GIVoxelPages& pages,
											  // Cache System
											  const GIVoxelCache& caches,
											  // Constants
											  uint32_t batchCount,
											  // Light Injection Related
											  const LightInjectParameters& injectParams,
											  const IEVector3& ambientColor,
											  bool injectOn)
{
	shadowMaps.Map();

	CudaTimer t;
	if(doTiming) t.Start();

	// Gen LI Params
	CLightInjectParameters liParams =
	{
		float3{ambientColor[0], ambientColor[1], ambientColor[2]},

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

	// KC
	uint32_t totalCount = pages.PageCount() * GIVoxelPages::PageSize;
	int gridSize = CudaInit::GenBlockSize(static_cast<int>(totalCount));
	int blockSize = CudaInit::TBP;
	SVOReconstruct<<<gridSize, blockSize>>>(// SVO
											dOctreeLevels,
											dLevelSizes,
											dLevelCapacities,
											// Voxel Pages
											pages.getVoxelPagesDevice(),
											pages.getVoxelGridsDevice(),
											// Cache System
											caches.getDeviceCascadePointersDevice().Data(),
											// LightInjectRelated
											liParams,
											// Limits
											*octreeParams,
											batchCount);
	CUDA_KERNEL_CHECK();

	// Recieve level sizess
	CUDA_CHECK(cudaMemcpy(hLevelSizes.data(), dLevelSizes,
						  (octreeParams->MaxSVOLevel + 1) * sizeof(uint32_t),
						  cudaMemcpyDeviceToHost));

	shadowMaps.Unmap();

	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}

	return 0.0;
}

double GISparseVoxelOctree::GenNeigbourPointers(bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();
	
	// Only Leaf node levels require this
	//for(uint32_t i = octreeParams->MaxSVOLevel - 1; i >= octreeParams->CascadeBaseLevel - 1; i--)
	for(uint32_t i = octreeParams->CascadeBaseLevel - 1; i < octreeParams->MaxSVOLevel; i++)
	{
		// Kernel Call Pointer Generation
		int gridSize = CudaInit::GenBlockSize(static_cast<int>(hLevelSizes[i]));
		int blockSize = CudaInit::TBP;

		// KC
		GenNeigbourPtrs<<<gridSize, blockSize>>>(// SVO
												 dOctreeLevels,
												 dLevelSizes,
												 dLevelCapacities,
												 // Limits
												 *octreeParams,
												 hLevelSizes[i],
												 i);
		CUDA_KERNEL_CHECK();
	}

	// Now We Created previous neigbours if not available
	// However these empty nodes may be in forward of other valid nodes
	// We need to check that

	// Recieve Used Level Sizes
	std::vector<uint32_t> extraLevelSizes(octreeParams->MaxSVOLevel + 1, 0);
	CUDA_CHECK(cudaMemcpy(extraLevelSizes.data(), dLevelSizes,
						  (octreeParams->MaxSVOLevel + 1) * sizeof(uint32_t),
						  cudaMemcpyDeviceToHost));

	// Make sure everything is connected
	for(uint32_t i = octreeParams->DenseLevel; i >= octreeParams->DenseLevel; i--)
	{
		// Kernel Call Light Injection
		int gridSize = CudaInit::GenBlockSize(extraLevelSizes[i]);
		int blockSize = CudaInit::TBP;

		// KC
		LinkNeigbourPtrs<<<gridSize, blockSize>>>(// SVO
												  dOctreeLevels,
												  dLevelSizes,
												  dLevelCapacities,
												  // Limits
												  *octreeParams,
												  extraLevelSizes[i],
												  i);
		CUDA_KERNEL_CHECK();
	}

	// Push Level Sizes
	hLevelSizes = extraLevelSizes;

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

	// Adjust Illum Nodes for averaging
	for(uint32_t i = octreeParams->MaxSVOLevel; i >= octreeParams->CascadeBaseLevel; i--)
	{
		// Kernel Call Light Injection
		int gridSize = CudaInit::GenBlockSize(static_cast<int>(hLevelSizes[i]));
		int blockSize = CudaInit::TBP;

		// KC
		AdjustIllumParameters<<<gridSize, blockSize>>>(dOctreeLevels[i], hLevelSizes[i]);
		CUDA_KERNEL_CHECK();
	}

	// Average Down to Top Fashion
	for(uint32_t i = octreeParams->MaxSVOLevel - 1; i >= octreeParams->DenseLevel; i--)
	{
		int denseLevelSize = (0x1 << i) * (0x1 << i) * (0x1 << i);
		int levelSize = (i == octreeParams->DenseLevel) ? denseLevelSize : hLevelSizes[i];
		int gridSize = CudaInit::GenBlockSize(static_cast<int>(levelSize * 2));
		int blockSize = CudaInit::TBP;

		const CSVOLevelConst& dNextLevel = reinterpret_cast<const CSVOLevelConst&>(dOctreeLevels[i + 1]);

		// KC
		AverageLevelSparse<<<gridSize, blockSize>>>(// SVO
													dOctreeLevels[i],
													dNextLevel,
													// Limits
													*octreeParams,
													static_cast<uint32_t>(levelSize),
													i >= octreeParams->CascadeBaseLevel);
	}
	for(uint32_t i = octreeParams->DenseLevel - 1; i >= octreeParams->MinSVOLevel; i--)
	{
		// Reset Base values (if there is a node)
		int levelSize = (0x1 << i) * (0x1 << i) * (0x1 << i);
		int gridSize = CudaInit::GenBlockSize(static_cast<int>(levelSize * 2));
		int blockSize = CudaInit::TBP;

		const CSVOLevelConst& dNextLevel = reinterpret_cast<const CSVOLevelConst&>(dOctreeLevels[i + 1]);

		// KC
		AverageLevelDense<<<gridSize, blockSize>>>(// SVO
												   dOctreeLevels[i],
												   dNextLevel,
												   // Limits
												   *octreeParams,
												   0x1 << i);
	}

	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

void GISparseVoxelOctree::UpdateSVO(// Timing Related
									double& reconstructTime,
									double& genPtrTime,
									double& averageTime,
									bool doTiming,
									// Page System
									const GIVoxelPages& pages,
									// Cache System
									const GIVoxelCache& caches,
									// Constants
									uint32_t batchCount,
									const LightInjectParameters& injectParams,
									const IEVector3& ambientColor,
									bool injectOn)
{
	MapOGLData();
	reconstructTime = GenerateHierarchy(doTiming, pages, caches, 
										batchCount, injectParams,
										ambientColor, injectOn);
	averageTime = AverageNodes(doTiming);
	genPtrTime = GenNeigbourPointers(doTiming);
	
	UnmapOGLData();
}

void GISparseVoxelOctree::UpdateOctreeUniforms(const IEVector3& outerCascadePos)
{
	// Octree Uniforms
	OctreeUniforms u = {};
	u.worldPos = outerCascadePos;
	u.baseSpan = octreeParams->BaseSpan;
	u.minSVOLevel = octreeParams->MinSVOLevel;
	u.denseLevel = octreeParams->DenseLevel;
	u.minCascadeLevel = octreeParams->CascadeBaseLevel;
	u.maxSVOLevel = octreeParams->MaxSVOLevel;
	u.cascadeCount = octreeParams->CascadeCount;
	u.nodeOffsetDifference = static_cast<uint32_t>(nodeIllumDifference);
	u.gridSize = octreeParams->CascadeBaseLevelSize;
	u.pad0 = 0xFFFFFFFF;
		
	oglData.SendSubData(reinterpret_cast<const uint8_t*>(&u),
						static_cast<uint32_t>(octreeUniformsOffset), 
						sizeof(OctreeUniforms));
}

void GISparseVoxelOctree::UpdateIndirectUniforms(const IndirectUniforms& indirectUniforms)
{
	std::memcpy(oglData.CPUData().data() + indirectUniformsOffset,
				&indirectUniforms,
				sizeof(IndirectUniforms));
	oglData.SendSubData(static_cast<uint32_t>(indirectUniformsOffset), sizeof(IndirectUniforms));
}

double GISparseVoxelOctree::GlobalIllumination(ConeTraceTexture& coneTex,
											   const DeferredRenderer& dRenderer,
											   const Camera& camera,
											   const IndirectUniforms&,
											   bool giOn,
											   bool aoOn,
											   bool specularOn)
{
	// Light Intensity Texture
	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glClearTexImage(coneTex.Texture(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);
	
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
	oglData.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET,
									  static_cast<uint32_t>(illumOffsetsOffset),
									  sizeof(uint32_t) * (octreeParams->MaxSVOLevel + 1));
	oglData.BindAsShaderStorageBuffer(LU_SVO_NODE, static_cast<uint32_t>(nodeOffset),
									  static_cast<uint32_t>(illumOffset - nodeOffset));
	oglData.BindAsShaderStorageBuffer(LU_SVO_ILLUM, static_cast<uint32_t>(illumOffset),
									  static_cast<uint32_t>(oglData.Count() - illumOffset));

	// Textures
	dRenderer.getGBuffer().BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	dRenderer.getGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	dRenderer.getGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);

	// Images
	glBindImageTexture(I_OUT_TEXTURE, coneTex.Texture(), 0, false, 0, GL_WRITE_ONLY, coneTex.Format());
	
	// Dispatch
	GLuint gridX = (coneTex.Width() + 16 - 1) / 16;
	GLuint gridY = (coneTex.Height() + 16 - 1) / 16;
	glDispatchCompute(gridX, gridY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	
	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);
	
	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
}

double GISparseVoxelOctree::DebugTraceSVO(ConeTraceTexture& coneTex,
										  const DeferredRenderer& dRenderer,
										  const Camera& camera,
										  uint32_t renderLevel,
										  OctreeRenderType octreeRender)
{

	// Light Intensity Texture
	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glClearTexImage(coneTex.Texture(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	// Shaders
	compVoxSampleWorld.Bind();

	// Uniforms
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(octreeRender));
	glUniform1ui(U_FETCH_LEVEL, renderLevel);

	// Uniform Buffers
	// Frame transform already bound
	dRenderer.BindInvFrameTransform(U_INVFTRANSFORM);
	oglData.BindAsUniformBuffer(U_OCTREE_UNIFORMS, static_cast<uint32_t>(octreeUniformsOffset),
								sizeof(OctreeUniforms));
	//oglData.BindAsUniformBuffer(U_INDIRECT_UNIFORMS, static_cast<uint32_t>(indirectUniformsOffset),
	//							sizeof(IndirectUniforms));

	// SSBO Buffers
	oglData.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET,
									  static_cast<uint32_t>(illumOffsetsOffset),
									  sizeof(uint32_t) * (octreeParams->MaxSVOLevel + 1));
	oglData.BindAsShaderStorageBuffer(LU_SVO_NODE, static_cast<uint32_t>(nodeOffset),
									  static_cast<uint32_t>(illumOffset - nodeOffset));
	oglData.BindAsShaderStorageBuffer(LU_SVO_ILLUM, static_cast<uint32_t>(illumOffset),
									  static_cast<uint32_t>(oglData.Count() - illumOffset));

	// Textures
	//dRenderer.getGBuffer().BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	dRenderer.getGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	//dRenderer.getGBuffer().BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);

	// Images
	glBindImageTexture(I_OUT_TEXTURE, coneTex.Texture(), 0, false, 0, GL_WRITE_ONLY, coneTex.Format());

	// Dispatch
	GLuint gridX = (coneTex.Width() + 16 - 1) / 16;
	GLuint gridY = (coneTex.Height() + 16 - 1) / 16;
	glDispatchCompute(gridX, gridY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);

	// I have to unbind the compute shader or weird things happen
	Shader::Unbind(ShaderType::COMPUTE);
	return timeElapsed / 1000000.0;
}

double GISparseVoxelOctree::DebugSampleSVO(ConeTraceTexture& coneTex,
										   const DeferredRenderer& dRenderer,
										   const Camera& camera,
										   uint32_t renderLevel,
										   OctreeRenderType octreeRender)
{
	// Light Intensity Texture
	static const GLubyte ff[4] = {0xFF, 0xFF, 0xFF, 0xFF};
	glClearTexImage(coneTex.Texture(), 0, GL_RGBA, GL_UNSIGNED_BYTE, &ff);

	// Timing Voxelization Process
	GLuint queryID;
	glGenQueries(1, &queryID);
	glBeginQuery(GL_TIME_ELAPSED, queryID);

	// Shaders
	compVoxSampleWorld.Bind();

	// Uniforms
	glUniform1ui(U_RENDER_TYPE, static_cast<GLuint>(octreeRender));
	glUniform1ui(U_FETCH_LEVEL, renderLevel);

	// Uniform Buffers
	// Frame transform already bound
	dRenderer.BindFrameTransform(U_FTRANSFORM);
	dRenderer.BindInvFrameTransform(U_INVFTRANSFORM);
	oglData.BindAsUniformBuffer(U_OCTREE_UNIFORMS, static_cast<uint32_t>(octreeUniformsOffset),
								sizeof(OctreeUniforms));

	// SSBO Buffers
	oglData.BindAsShaderStorageBuffer(LU_SVO_LEVEL_OFFSET,
									  static_cast<uint32_t>(illumOffsetsOffset),
									  sizeof(uint32_t) * (octreeParams->MaxSVOLevel + 1));
	oglData.BindAsShaderStorageBuffer(LU_SVO_NODE, static_cast<uint32_t>(nodeOffset),
									  static_cast<uint32_t>(illumOffset - nodeOffset));
	oglData.BindAsShaderStorageBuffer(LU_SVO_ILLUM, static_cast<uint32_t>(illumOffset),
									  static_cast<uint32_t>(oglData.Count() - illumOffset));

	// Textures
	//dRenderer.getGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::COLOR);
	dRenderer.getGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	//dRenderer.getGBuffer().BindAsTexture(T_DEPTH, RenderTargetLocation::NORMAL);

	// Images
	glBindImageTexture(I_OUT_TEXTURE, coneTex.Texture(), 0, false, 0, GL_WRITE_ONLY, coneTex.Format());

	// Dispatch
	GLuint gridX = (coneTex.Width() + 16 - 1) / 16;
	GLuint gridY = (coneTex.Height() + 16 - 1) / 16;
	glDispatchCompute(gridX, gridY, 1);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	// Timer
	GLuint64 timeElapsed = 0;
	glEndQuery(GL_TIME_ELAPSED);
	glGetQueryObjectui64v(queryID, GL_QUERY_RESULT, &timeElapsed);

	return timeElapsed / 1000000.0;
}

size_t GISparseVoxelOctree::MemoryUsage() const
{
	return (oglData.Capacity() + cudaData.Size());
}