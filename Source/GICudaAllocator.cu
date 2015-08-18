#include "GICudaAllocator.h"
#include "GICudaStructMatching.h"
#include <cuda_gl_interop.h>
#include "CudaTimer.h"
#include "Macros.h"

// Small Helper Kernel That used to init inital obj Pages
// Logic is per segment
__global__ void EmptyPageInit(unsigned int* gPageEmptySegmentPos)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= GI_SEGMENT_PER_PAGE) return;
	gPageEmptySegmentPos[globalId] = globalId;
}

__global__ void SegmentAllocLocInit(ushort2* gSegments,
									const uint32_t segmentCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= segmentCount) return;
	gSegments[globalId].x = 0xFFFF;
	gSegments[globalId].y = 0xFFFF;
}

// Small Helper Kernel That used to determine total segment size used by the object batch
// Logic per object in batch
__global__ void DetermineTotalSegment(int* dTotalSegmentCount,

									  // Grid Related
									  const CVoxelGrid& gGridInfo,

									  // Per object Related
									  unsigned int* gObjectVoxStrides,
									  unsigned int* gObjectAllocIndexLookup,
									  const CObjectVoxelInfo* gVoxelInfo,
									  const CObjectTransform* gObjTransforms,
									  const uint32_t objCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;

	// Determine Segment Count and add do total segment counter

	// We need to check scaling and adjust span
	// Objects may have different voxel sizes and voxel sizes may change after scaling
	float3 scaling = ExtractScaleInfo(gObjTransforms[globalId].transform);
	assert(scaling.x == scaling.y);
	assert(scaling.y == scaling.z);

	unsigned int voxelDim = static_cast<unsigned int>(gVoxelInfo[globalId].span * scaling.x / gGridInfo.span);
	unsigned int voxScale = voxelDim == 0 ? 0 : 1;
	unsigned int segmentCount = ((gVoxelInfo[globalId].voxelCount * voxScale) + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
	
	// Determine Strides
	// Here this implementation is slow and does redundant work 
	// but its most easily written version
	unsigned int objStirde = 0, objIndexLookup = 0;
	for(unsigned int i = 0; i < globalId; i++)
	{
		float3 scalingObj = ExtractScaleInfo(gObjTransforms[i].transform);
		assert(scalingObj.x == scalingObj.y);
		assert(scalingObj.y == scalingObj.z);

		unsigned int voxelDim = static_cast<unsigned int>(gVoxelInfo[i].span * scaling.x / gGridInfo.span);
		unsigned int voxScaleObj = voxelDim == 0 ? 0 : 1;
		
		objStirde += gVoxelInfo[i].voxelCount * voxScaleObj;
		objIndexLookup += ((gVoxelInfo[i].voxelCount * voxScaleObj) + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
	}

	if(globalId == objCount - 1)
		dTotalSegmentCount[0] = objIndexLookup + segmentCount;

	gObjectVoxStrides[globalId] = objStirde;
	gObjectAllocIndexLookup[globalId] = objIndexLookup;
}

// Used to populate segment object id's
// Logic per object in batch
__global__ void DetermineSegmentObjId(unsigned int* gSegmentObjectId,

									  // Grid Related
									  const CVoxelGrid& gGridInfo,

									  const unsigned int* gObjectAllocIndexLookup,
									  const CObjectVoxelInfo* gVoxelInfo,
									  const CObjectTransform* gObjTransforms,
									  const uint32_t objCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;

	float3 scaling = ExtractScaleInfo(gObjTransforms[globalId].transform);
	assert(scaling.x == scaling.y);
	assert(scaling.y == scaling.z);
	unsigned int voxelDim = static_cast<unsigned int>(gVoxelInfo[globalId].span * scaling.x / gGridInfo.span);
	unsigned int voxScale = voxelDim == 0 ? 0 : 1;

	unsigned int segmentCount = ((gVoxelInfo[globalId].voxelCount * voxScale) + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
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
	dVoxelGridInfo.Assign(0, hVoxelGridInfo);
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
	cudaError_t cudaErr;
	CudaTimer timer(0);
	timer.Start();

	rTransformLinks.emplace_back();
	transformLinks.emplace_back();
	aabbLinks.emplace_back();
	objectInfoLinks.emplace_back();
	cacheLinks.emplace_back();
	cacheRenderLinks.emplace_back();

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

	dVoxelStrides.back().Memset(0, 0, dVoxelStrides.back().Size());
	dObjectAllocationIndexLookup.back().Memset(0, 0, dObjectAllocationIndexLookup.back().Size());
	dWriteSignals.back().Memset(0, 0, dWriteSignals.back().Size());
	
	// Populate Helper Data
	// Determine object segement sizes
	int* dTotalCount = nullptr;
	int hTotalCount = 0;
	cudaErr = cudaMalloc(reinterpret_cast<void**>(&dTotalCount), sizeof(int));
	cudaErr = cudaMemcpy(dTotalCount, &hTotalCount, sizeof(int), cudaMemcpyHostToDevice);

	// Mapping Pointer
	CObjectVoxelInfo* dVoxelInfo = nullptr;
	CObjectTransform* dObjTransform = nullptr;
	size_t size = 0;
	cudaGraphicsMapResources(1, &objectInfoLinks.back());
	cudaGraphicsMapResources(1, &transformLinks.back());
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxelInfo), &size, objectInfoLinks.back());
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dObjTransform), &size, transformLinks.back());
	
	uint32_t gridSize = static_cast<uint32_t>((objCount + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK);
	DetermineTotalSegment<<<gridSize, GI_THREAD_PER_BLOCK>>>(dTotalCount,
															 *dVoxelGridInfo.Data(),

															 dVoxelStrides.back().Data(),
															 dObjectAllocationIndexLookup.back().Data(),
															 dVoxelInfo,
															 dObjTransform,
															 objCount);
	
	// Allocation after determining total index count
	cudaMemcpy(&hTotalCount, dTotalCount, sizeof(int), cudaMemcpyDeviceToHost);
	dSegmentObjecId.emplace_back(hTotalCount);
	dSegmentAllocLoc.emplace_back(hTotalCount);
	dSegmentAllocLoc.back().Memset(0xFF, 0, dSegmentAllocLoc.back().Size());

	gridSize = static_cast<uint32_t>((objCount + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK);
	DetermineSegmentObjId<<<gridSize, GI_THREAD_PER_BLOCK>>>(dSegmentObjecId.back().Data(),
															 *dVoxelGridInfo.Data(),

															 dObjectAllocationIndexLookup.back().Data(),
															 dVoxelInfo,
															 dObjTransform,
															 objCount);
	cudaGraphicsUnmapResources(1, &objectInfoLinks.back());
	cudaGraphicsMapResources(1, &transformLinks.back());

	dObjectAllocationIndexLookup2D.InsertEnd(dObjectAllocationIndexLookup.back().Data());
	dSegmentAllocLoc2D.InsertEnd(dSegmentAllocLoc.back().Data());
	dObjectVoxStrides2D.InsertEnd(dVoxelStrides.back().Data());
	cudaFree(dTotalCount);
	timer.Stop();
	GI_LOG("Linked Object Batch to CUDA. Elaped time %f ms", timer.ElapsedMilliS());

	// Dump Initial Helper Files
	//dSegmentAllocLoc.back().DumpToFile("segAllocLoc");
	//dSegmentObjecId.back().DumpToFile("segObjId");

	//dVoxelStrides.back().DumpToFile("voxStride");
	//dObjectAllocationIndexLookup.back().DumpToFile("allocIndexLookup");
	//dWriteSignals.back().DumpToFile("writeSignals");

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

void GICudaAllocator::LinkSceneShadowMapArray(GLuint shadowMapArray)
{
	cudaError_t cudaErr;
	cudaErr = cudaGraphicsGLRegisterImage(&sceneShadowMapLink,
										  shadowMapArray,
										  GL_TEXTURE_2D_ARRAY,
										  cudaGraphicsRegisterFlagsReadOnly);
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

	//cudaError_t cerr;
	//cerr = cudaGraphicsMapResources(1, &sceneShadowMapLink);
	//cerr = cudaGraphicsResourceGetMappedMipmappedArray(&mipArray, sceneShadowMapLink);
	//resDesc.res.mipmap.mipmap = mipArray;
	//cerr = cudaCreateTextureObject(&shadowMaps, &resDesc, &texDesc, nullptr);

	//texDesc.normalizedCoords = 1;
	//resDesc.resType = cudaResourceTypeArray;

	//cerr = cudaGraphicsMapResources(1, &depthBuffLink);
	//cerr = cudaGraphicsSubResourceGetMappedArray(&texArray, depthBuffLink, 0, 0);
	//resDesc.res.array.array = texArray;
	//cerr = cudaCreateTextureObject(&depthBuffer, &resDesc, &texDesc, nullptr);

	//cerr = cudaGraphicsMapResources(1, &normalBuffLink);
	//cudaGraphicsSubResourceGetMappedArray(&texArray, normalBuffLink, 0, 0);
	//resDesc.res.array.array = texArray;
	//cudaCreateTextureObject(&normalBuffer, &resDesc, &texDesc, nullptr);

	//cudaGraphicsMapResources(1, &lightIntensityLink);
	//cudaGraphicsSubResourceGetMappedArray(&texArray, lightIntensityLink, 0, 0);
	//resDesc.res.array.array = texArray;
	//cudaCreateSurfaceObject(&lightIntensityBuffer, &resDesc);
}

void GICudaAllocator::ClearDevicePointers()
{
	dRelativeTransforms.Clear();
	dTransforms.Clear();
	dObjectAABB.Clear();
	dObjectInfo.Clear();

	dObjCache.Clear();
	dObjRenderCache.Clear();

	hRelativeTransforms.clear();
	hTransforms.clear();
	hObjectAABB.clear();
	hObjectInfo.clear();

	hObjCache.clear();
	hObjRenderCache.clear();
	
	cudaError_t cerr;
	//cerr = cudaDestroySurfaceObject(lightIntensityBuffer);
	//cerr = cudaDestroyTextureObject(normalBuffer);
	//cerr = cudaDestroyTextureObject(depthBuffer);
	//cerr = cudaDestroyTextureObject(shadowMaps);

	//cerr = cudaGraphicsUnmapResources(1, &lightIntensityLink);
	//cerr = cudaGraphicsUnmapResources(1, &normalBuffLink);
	//cerr = cudaGraphicsUnmapResources(1, &depthBuffLink);
	//cerr = cudaGraphicsUnmapResources(1, &sceneShadowMapLink);

	cerr = cudaGraphicsUnmapResources(static_cast<int>(rTransformLinks.size()), rTransformLinks.data());
	cerr = cudaGraphicsUnmapResources(static_cast<int>(transformLinks.size()), transformLinks.data());
	cerr = cudaGraphicsUnmapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data());
	cerr = cudaGraphicsUnmapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data());

	cerr = cudaGraphicsUnmapResources(static_cast<int>(cacheLinks.size()), cacheLinks.data());
	cerr = cudaGraphicsUnmapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data());
}

void GICudaAllocator::AddVoxelPage(size_t count)
{
	hPageData.reserve(hPageData.size() + count);
	for(unsigned int i = 0; i < count; i++)
	{
		// Allocating Page
		hPageData.emplace_back(GI_PAGE_SIZE, GI_SEGMENT_PER_PAGE);
		EmptyPageInit<<<(GI_SEGMENT_PER_PAGE + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK, GI_THREAD_PER_BLOCK>> >
		(
			hPageData.back().dEmptySegmentList.Data()
		);
		hPageData.back().dIsSegmentOccupied.Memset(0, 0, hPageData.back().dIsSegmentOccupied.Size());
		hPageData.back().dVoxelPage.Memset(0, 0, hPageData.back().dVoxelPage.Size());

		CVoxelPage voxData =
		{
			hPageData.back().dVoxelPage.Data(),
			hPageData.back().dEmptySegmentList.Data(),
			hPageData.back().dIsSegmentOccupied.Data(),
			GI_SEGMENT_PER_PAGE
		};
		hVoxelPages.push_back(voxData);

		//if(i == 0)
		//{
		//	hPageData.back().dEmptySegmentList.DumpToFile("pageEmpty");
		//	hPageData.back().dIsSegmentOccupied.DumpToFile("pageOccp");
		//}
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
	cudaGraphicsUnregisterResource(sceneShadowMapLink);

	rTransformLinks.clear();
	transformLinks.clear();
	aabbLinks.clear();
	objectInfoLinks.clear();

	cacheLinks.clear();
	cacheRenderLinks.clear();

	dSegmentObjecId.clear();
	dSegmentAllocLoc.clear();

	dVoxelStrides.clear();
	dObjectAllocationIndexLookup.clear();
	dWriteSignals.clear();
	
	dObjectAllocationIndexLookup2D.Clear();
	dSegmentAllocLoc2D.Clear();

	objectCounts.clear();
	voxelCounts.clear();

	totalObjectCount = 0;
}

void GICudaAllocator::Reserve(uint32_t pageAmount)
{
	if(dVoxelPages.Size() < pageAmount)
	{
		AddVoxelPage(pageAmount - dVoxelPages.Size());
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
	return static_cast<uint32_t>(dSegmentObjecId[batchIndex].Size());
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
	return dVoxelGridInfo.Data();
}

CVoxelGrid GICudaAllocator::GetVoxelGridHost()
{
	return hVoxelGridInfo;
}

IEVector3 GICudaAllocator::GetNewVoxelPos(const IEVector3& playerPos)
{
	hVoxelGridInfo.position.x = playerPos.getX() - hVoxelGridInfo.span * hVoxelGridInfo.dimension.x * 0.5f;
	hVoxelGridInfo.position.y = playerPos.getY() - hVoxelGridInfo.span * hVoxelGridInfo.dimension.y * 0.5f;
	hVoxelGridInfo.position.z = playerPos.getZ() - hVoxelGridInfo.span * hVoxelGridInfo.dimension.z * 0.5f;

	//TODO: Dummy update, remove this when you call actual update
	dVoxelGridInfo.Assign(0, hVoxelGridInfo);
	
	return {hVoxelGridInfo.position.x, hVoxelGridInfo.position.y, hVoxelGridInfo.position.z};
}

CObjectTransform** GICudaAllocator::GetRelativeTransformsDevice() 
{
	return dRelativeTransforms.Data();
}

CObjectTransform** GICudaAllocator::GetTransformsDevice()
{
	return dTransforms.Data();
}

CObjectAABB** GICudaAllocator::GetObjectAABBDevice()
{
	return dObjectAABB.Data();
}

CObjectVoxelInfo** GICudaAllocator::GetObjectInfoDevice()
{
	return dObjectInfo.Data();
}

CVoxelPacked** GICudaAllocator::GetObjCacheDevice()
{
	return dObjCache.Data();
}

CVoxelRender** GICudaAllocator::GetObjRenderCacheDevice()
{
	return dObjRenderCache.Data();
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
	return dVoxelPages.Data();
}

unsigned int* GICudaAllocator::GetSegmentObjectID(uint32_t index)
{
	return dSegmentObjecId[index].Data();
}

ushort2* GICudaAllocator::GetSegmentAllocLoc(uint32_t index)
{
	return dSegmentAllocLoc[index].Data();
}

unsigned int* GICudaAllocator::GetVoxelStrides(uint32_t index)
{
	return dVoxelStrides[index].Data();
}

unsigned int* GICudaAllocator::GetObjectAllocationIndexLookup(uint32_t index)
{
	return dObjectAllocationIndexLookup[index].Data();
}

char* GICudaAllocator::GetWriteSignals(uint32_t index)
{
	return dWriteSignals[index].Data();
}

unsigned int** GICudaAllocator::GetObjectAllocationIndexLookup2D()
{
	return dObjectAllocationIndexLookup2D.Data();
}

unsigned int** GICudaAllocator::GetObjectVoxStrides2D()
{
	return dObjectVoxStrides2D.Data();
}

ushort2** GICudaAllocator::GetSegmentAllocLoc2D()
{
	return dSegmentAllocLoc2D.Data();
}