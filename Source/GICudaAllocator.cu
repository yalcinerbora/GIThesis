#include "GICudaAllocator.h"
#include "GICudaStructMatching.h"
#include <cuda_gl_interop.h>
#include "CudaTimer.h"
#include "Macros.h"

// Small Helper Kernel That used to determine total segment size used by the object batch
// Logic per object in batch
__global__ void DetermineTotalSegment(int& dTotalSegmentCount,

									  // Per object Related
									  unsigned int* gObjectVoxStrides,
									  unsigned int* gObjectAllocIndexLookup,
									  const CObjectVoxelInfo* gVoxelInfo,
									  uint32_t objCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;

	// Determine Segment Count and add do total segment counter
	unsigned int segmentCount = (gVoxelInfo[globalId].voxelCount + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
	atomicAdd(&dTotalSegmentCount, segmentCount);

	// Determine Strides
	// Here this implementation is slow and does redundant work 
	// but its most easily written version
	unsigned int objStirde = 0, objIndexLookup = 0;
	for(unsigned int i = 0; i < globalId; i++)
	{
		objStirde += gVoxelInfo[i].voxelCount;
		unsigned int segmentCount = (gVoxelInfo[i].voxelCount + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
		objIndexLookup += segmentCount;
	}

	gObjectVoxStrides[globalId] = objStirde;
	gObjectAllocIndexLookup[globalId] = objIndexLookup;
}

// Used to populate segment object id's
// Logic per object in batch
__global__ void DetermineSegmentObjId(unsigned int* gSegmentObjectId,

									  const unsigned int* gObjectAllocIndexLookup,
									  const CObjectVoxelInfo* gVoxelInfo,
									  uint32_t objCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;

	unsigned int segmentCount = (gVoxelInfo[globalId].voxelCount + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
	for(unsigned int i = 0; i < segmentCount; i++)
	{
		gSegmentObjectId[gObjectAllocIndexLookup[globalId] + i] = globalId;
	}
}


GICudaAllocator::GICudaAllocator(const CVoxelGrid& gridInfo)
	: totalObjectCount(0)
	, dVoxelGridInfo(1)
	, hVoxelGridInfo(gridInfo)
{
	cudaGLSetGLDevice(0);
	dVoxelGridInfo[0] = hVoxelGridInfo;
}

void GICudaAllocator::LinkOGLVoxelCache(GLuint batchAABBBuffer,
										GLuint batchTransformBuffer,
										GLuint relativeTransformBuffer,
										GLuint infoBuffer,
										GLuint voxelBuffer,
										GLuint voxelRenderBuffer,
										uint32_t objCount,
										uint32_t voxelCount)
{
	CudaTimer timer(0);
	timer.Start();

	rTransformLinks.emplace_back(nullptr);
	transformLinks.emplace_back(nullptr);
	aabbLinks.emplace_back(nullptr);
	objectInfoLinks.emplace_back(nullptr);
	cacheLinks.emplace_back(nullptr);
	cacheRenderLinks.emplace_back(nullptr);

	cudaGraphicsGLRegisterBuffer(&rTransformLinks.back(), relativeTransformBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&transformLinks.back(), batchTransformBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&aabbLinks.back(), batchAABBBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&objectInfoLinks.back(), infoBuffer, cudaGraphicsMapFlagsReadOnly);

	cudaGraphicsGLRegisterBuffer(&cacheLinks.back(), voxelBuffer, cudaGraphicsMapFlagsReadOnly);
	cudaGraphicsGLRegisterBuffer(&cacheRenderLinks.back(), voxelRenderBuffer, cudaGraphicsMapFlagsReadOnly);

	objectCounts.emplace_back(objCount);
	voxelCounts.emplace_back(voxelCount);
	totalObjectCount += objCount;

	// Allocate Helper Data
	dVoxelStrides.emplace_back(objCount);
	dObjectAllocationIndexLookup.emplace_back(objCount);
	dWriteSignals.emplace_back(objCount);

	// Populate Helper Data
	// Determine object segement sizes
	int* dTotalCount = nullptr;
	int hTotalCount = 0;
	cudaMalloc(reinterpret_cast<void**>(&dTotalCount), sizeof(int));
	cudaMemset(dTotalCount, 0, sizeof(int));

	// Mapping Pointer
	CObjectVoxelInfo* dVoxelInfo = nullptr;
	size_t size = 0;
	cudaGraphicsMapResources(1, &objectInfoLinks.back());
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxelInfo), &size, objectInfoLinks.back());
	
	uint32_t gridSize = static_cast<uint32_t>((objCount + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK);
	DetermineTotalSegment<<<gridSize, GI_THREAD_PER_BLOCK>>>(*dTotalCount,
															 thrust::raw_pointer_cast(dVoxelStrides.back().data()),
															 thrust::raw_pointer_cast(dObjectAllocationIndexLookup.back().data()),
															 dVoxelInfo,
															 objCount);

	cudaMemcpy(&hTotalCount, dTotalCount, sizeof(int), cudaMemcpyDeviceToHost);
	dSegmentObjecId.emplace_back(hTotalCount);
	dSegmentAllocLoc.emplace_back(hTotalCount);

	DetermineSegmentObjId<<<gridSize, GI_THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(dSegmentObjecId.back().data()),
															 thrust::raw_pointer_cast(dObjectAllocationIndexLookup.back().data()),
															 dVoxelInfo,
															 objCount);

	cudaGraphicsUnmapResources(1, &objectInfoLinks.back());

	dObjectAllocationIndexLookup2D.push_back(thrust::raw_pointer_cast(dObjectAllocationIndexLookup.back().data()));
	dSegmentAllocLoc2D.push_back(thrust::raw_pointer_cast(dSegmentAllocLoc.back().data()));

	timer.Stop();
	GI_LOG("Linked Object Batch to CUDA. Elaped time %f ms", timer.ElapsedMilliS());

	assert(rTransformLinks.size() == transformLinks.size());
	assert(transformLinks.size() == aabbLinks.size());
	assert(aabbLinks.size() == transformLinks.size());
	assert(rTransformLinks.size() == objectInfoLinks.size());
	assert(objectInfoLinks.size() == cacheLinks.size());
	assert(cacheLinks.size() == cacheRenderLinks.size());
	assert(cacheRenderLinks.size() == dVoxelStrides.size());
	assert(dVoxelStrides.size() == dObjectAllocationIndexLookup.size());
	assert(dObjectAllocationIndexLookup.size() == dWriteSignals.size());
	assert(dWriteSignals.size() == dSegmentObjecId.size());
	assert(dSegmentObjecId.size() == dSegmentAllocLoc.size());
}

void GICudaAllocator::LinkSceneShadowMapArray(const std::vector<GLuint>& shadowMaps)
{
	cudaGraphicsResource* resource = nullptr;
	for(unsigned int i = 0; i < shadowMaps.size(); i++)
	{
		cudaGraphicsGLRegisterImage(&resource,
									shadowMaps[i],
									GL_TEXTURE_CUBE_MAP,
									cudaGraphicsRegisterFlagsReadOnly);
		sceneShadowMapLinks.push_back(resource);
	}
}

void GICudaAllocator::LinkSceneGBuffers(GLuint depthTex,
										GLuint normalTex,
										GLuint lightIntensityTex)
{
	cudaGraphicsGLRegisterImage(&depthBuffLink,
								depthTex,
								GL_TEXTURE_2D,
								cudaGraphicsRegisterFlagsReadOnly);
	cudaGraphicsGLRegisterImage(&normalBuffLink,
								normalTex,
								GL_TEXTURE_2D,
								cudaGraphicsRegisterFlagsReadOnly);
	cudaGraphicsGLRegisterImage(&lightIntensityLink,
								lightIntensityTex,
								GL_TEXTURE_2D,
								cudaGraphicsRegisterFlagsSurfaceLoadStore);
}

void GICudaAllocator::UnLinkGBuffers()
{
	cudaGraphicsUnregisterResource(depthBuffLink);
	cudaGraphicsUnregisterResource(normalBuffLink);
	cudaGraphicsUnregisterResource(lightIntensityLink);
}

void GICudaAllocator::SetupDevicePointers()
{
	cudaGraphicsMapResources(static_cast<int>(rTransformLinks.size()), rTransformLinks.data());
	cudaGraphicsMapResources(static_cast<int>(transformLinks.size()), transformLinks.data());
	cudaGraphicsMapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data());
	cudaGraphicsMapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data());

	cudaGraphicsMapResources(static_cast<int>(cacheLinks.size()), cacheLinks.data());
	cudaGraphicsMapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data());

	size_t size = 0;
	for(unsigned int i = 0; i < objectCounts.size(); i++)
	{
		hRelativeTransforms.push_back(nullptr);
		hTransforms.push_back(nullptr);
		hObjectAABB.push_back(nullptr);
		hObjectInfo.push_back(nullptr);

		hObjCache.push_back(nullptr);
		hObjRenderCache.push_back(nullptr);

		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hRelativeTransforms.back()), &size, rTransformLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hTransforms.back()), &size, transformLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjectAABB.back()), &size, aabbLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjectInfo.back()), &size, objectInfoLinks[i]);

		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjCache.back()), &size, cacheLinks[i]);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjRenderCache.back()), &size, cacheRenderLinks[i]);
	}

	//// Data Sent to GPU
	dRelativeTransforms = hRelativeTransforms;
	dTransforms = hTransforms;
	dObjectAABB = hObjectAABB;
	dObjectInfo = hObjectInfo;

	dObjCache = hObjCache;
	dObjRenderCache = hObjRenderCache;


	// Textures
	cudaArray* texArray = nullptr;
	cudaResourceDesc resDesc = {};
	cudaTextureDesc texDesc = {};

	resDesc.res.array.array = texArray;
	resDesc.resType = cudaResourceTypeArray;

	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 0;

	cudaGraphicsMapResources(static_cast<int>(sceneShadowMapLinks.size()), sceneShadowMapLinks.data());
	for(unsigned int i = 0; i < sceneShadowMapLinks.size(); i++)
	{
		cudaGraphicsSubResourceGetMappedArray(&texArray, sceneShadowMapLinks[i], 0, 0);

		shadowMaps.emplace_back();
		cudaCreateTextureObject(&shadowMaps.back(), &resDesc, &texDesc, nullptr);
	}

	cudaGraphicsMapResources(1, &depthBuffLink);
	cudaGraphicsSubResourceGetMappedArray(&texArray, depthBuffLink, 0, 0);
	texDesc.readMode = cudaReadModeElementType;
	cudaCreateTextureObject(&depthBuffer, &resDesc, &texDesc, nullptr);

	cudaGraphicsMapResources(1, &normalBuffLink);
	cudaGraphicsSubResourceGetMappedArray(&texArray, normalBuffLink, 0, 0);
	texDesc.readMode = cudaReadModeElementType;
	cudaCreateTextureObject(&normalBuffer, &resDesc, &texDesc, nullptr);

	cudaGraphicsMapResources(1, &lightIntensityLink);
	cudaGraphicsSubResourceGetMappedArray(&texArray, lightIntensityLink, 0, 0);
	cudaCreateSurfaceObject(&lightIntensityBuffer, &resDesc);

}

void GICudaAllocator::ClearDevicePointers()
{
	dRelativeTransforms.clear();
	dTransforms.clear();
	dObjectAABB.clear();
	dObjectInfo.clear();

	dObjCache.clear();
	dObjRenderCache.clear();

	hRelativeTransforms.clear();
	hTransforms.clear();
	hObjectAABB.clear();
	hObjectInfo.clear();

	hObjCache.clear();
	hObjRenderCache.clear();

	cudaDestroySurfaceObject(lightIntensityBuffer);
	cudaDestroyTextureObject(normalBuffer);
	cudaDestroyTextureObject(depthBuffer);

	for(unsigned int i = 0; i < shadowMaps.size(); i++)
	{
		cudaDestroyTextureObject(shadowMaps[i]);
	}
	shadowMaps.clear();

	cudaGraphicsUnmapResources(1, &depthBuffLink);
	cudaGraphicsUnmapResources(1, &normalBuffLink);
	cudaGraphicsUnmapResources(1, &lightIntensityLink);
	cudaGraphicsUnmapResources(static_cast<int>(sceneShadowMapLinks.size()), sceneShadowMapLinks.data());

	cudaGraphicsUnmapResources(static_cast<int>(rTransformLinks.size()), rTransformLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(transformLinks.size()), transformLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data());

	cudaGraphicsUnmapResources(static_cast<int>(cacheLinks.size()), cacheLinks.data());
	cudaGraphicsUnmapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data());
}

void GICudaAllocator::AddVoxelPage(size_t count)
{
	for(unsigned int i = 0; i < count; i++)
	{
		// Allocating Page
		hPageData.emplace_back(CVoxelPageData
		{
			thrust::device_vector<CVoxelPacked>(GI_PAGE_SIZE, CVoxelPacked {0, 0, 0, 0}),
			thrust::device_vector<unsigned int>(GI_BLOCK_PER_PAGE, 0),
			thrust::device_vector<char>(GI_BLOCK_PER_PAGE, 0)
		});

		CVoxelPage voxData =
		{
			thrust::raw_pointer_cast(hPageData.back().dVoxelPage.data()),
			thrust::raw_pointer_cast(hPageData.back().dEmptySegmentList.data()),
			thrust::raw_pointer_cast(hPageData.back().dIsSegmentOccupied.data()),
			0
		};
		hVoxelPages.push_back(voxData);
	}
	dVoxelPages = hVoxelPages;
}

void GICudaAllocator::ResetSceneData()
{
	for(unsigned int i = 0; i < rTransformLinks.size(); i++)
	{
		cudaGraphicsUnregisterResource(rTransformLinks[i]);
		cudaGraphicsUnregisterResource(transformLinks[i]);
		cudaGraphicsUnregisterResource(aabbLinks[i]);
		cudaGraphicsUnregisterResource(objectInfoLinks[i]);

		cudaGraphicsUnregisterResource(cacheLinks[i]);
		cudaGraphicsUnregisterResource(cacheRenderLinks[i]);
	}

	for(unsigned int i = 0; i < sceneShadowMapLinks.size(); i++)
	{
		cudaGraphicsUnregisterResource(sceneShadowMapLinks[i]);
	}

	rTransformLinks.clear();
	transformLinks.clear();
	aabbLinks.clear();
	objectInfoLinks.clear();

	cacheLinks.clear();
	cacheRenderLinks.clear();

	sceneShadowMapLinks.clear();

	dSegmentObjecId.clear();
	dSegmentAllocLoc.clear();

	dVoxelStrides.clear();
	dObjectAllocationIndexLookup.clear();
	dWriteSignals.clear();
	
	dObjectAllocationIndexLookup2D.clear();
	dSegmentAllocLoc2D.clear();

	objectCounts.clear();
	voxelCounts.clear();

	totalObjectCount = 0;
}

void GICudaAllocator::Reserve(uint32_t pageAmount)
{
	if(dVoxelPages.size() < pageAmount)
	{
		AddVoxelPage(pageAmount - dVoxelPages.size());
	}
}

uint32_t GICudaAllocator::NumObjectBatches() const
{
	return static_cast<uint32_t>(rTransformLinks.size());
}

uint32_t GICudaAllocator::NumObjects(uint32_t batchIndex) const
{
	return static_cast<uint32_t>(objectCounts[batchIndex]);
}

uint32_t GICudaAllocator::NumObjectSegments(uint32_t batchIndex) const
{
	return static_cast<uint32_t>(dSegmentObjecId[batchIndex].size());
}

uint32_t GICudaAllocator::NumVoxels(uint32_t batchIndex) const
{
	return static_cast<uint32_t>(voxelCounts[batchIndex]);
}

uint32_t GICudaAllocator::NumPages() const
{
	return static_cast<uint32_t>(hVoxelPages.size());
}

CVoxelGrid* GICudaAllocator::GetVoxelGridDevice()
{
	return thrust::raw_pointer_cast(dVoxelGridInfo.data());
}

CObjectTransform** GICudaAllocator::GetRelativeTransformsDevice() 
{
	return thrust::raw_pointer_cast(dRelativeTransforms.data());
}

CObjectTransform** GICudaAllocator::GetTransformsDevice()
{
	return thrust::raw_pointer_cast(dTransforms.data());
}

CObjectAABB** GICudaAllocator::GetObjectAABBDevice()
{
	return thrust::raw_pointer_cast(dObjectAABB.data());
}

CObjectVoxelInfo** GICudaAllocator::GetObjectInfoDevice()
{
	return thrust::raw_pointer_cast(dObjectInfo.data());
}

CVoxelPacked** GICudaAllocator::GetObjCacheDevice()
{
	return thrust::raw_pointer_cast(dObjCache.data());;
}

CVoxelRender** GICudaAllocator::GetObjRenderCacheDevice()
{
	return thrust::raw_pointer_cast(dObjRenderCache.data());
}

CObjectTransform* GICudaAllocator::GetRelativeTransformsDevice(uint32_t index)
{
	return hRelativeTransforms[index];
}

CObjectTransform* GICudaAllocator::GetTransformsDevice(uint32_t index)
{
	return hTransforms[index];
}

CObjectAABB* GICudaAllocator::GetObjectAABBDevice(uint32_t index)
{
	return hObjectAABB[index];
}

CObjectVoxelInfo* GICudaAllocator::GetObjectInfoDevice(uint32_t index)
{
	return hObjectInfo[index];
}


CVoxelPacked* GICudaAllocator::GetObjCacheDevice(uint32_t index)
{
	return hObjCache[index];
}

CVoxelRender* GICudaAllocator::GetObjRenderCacheDevice(uint32_t index)
{
	return hObjRenderCache[index];
}

CVoxelPage* GICudaAllocator::GetVoxelPagesDevice()
{
	return thrust::raw_pointer_cast(dVoxelPages.data());
}

unsigned int* GICudaAllocator::GetSegmentObjectID(uint32_t index)
{
	return thrust::raw_pointer_cast(dSegmentObjecId[index].data());
}

ushort2* GICudaAllocator::GetSegmentAllocLoc(uint32_t index)
{
	return thrust::raw_pointer_cast(dSegmentAllocLoc[index].data());
}

unsigned int* GICudaAllocator::GetVoxelStrides(uint32_t index)
{
	return thrust::raw_pointer_cast(dVoxelStrides[index].data());
}

unsigned int* GICudaAllocator::GetObjectAllocationIndexLookup(uint32_t index)
{
	return thrust::raw_pointer_cast(dObjectAllocationIndexLookup[index].data());
}

char* GICudaAllocator::GetWriteSignals(uint32_t index)
{
	return thrust::raw_pointer_cast(dWriteSignals[index].data());
}

unsigned int** GICudaAllocator::GetObjectAllocationIndexLookup2D()
{
	return thrust::raw_pointer_cast(dObjectAllocationIndexLookup2D.data());
}

ushort2** GICudaAllocator::GetSegmentAllocLoc2D()
{
	return thrust::raw_pointer_cast(dSegmentAllocLoc2D.data());
}