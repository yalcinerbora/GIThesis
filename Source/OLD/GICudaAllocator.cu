#include "GICudaAllocator.h"
#include "GICudaStructMatching.h"
#include <cuda_gl_interop.h>
#include "CudaTimer.h"
#include "Macros.h"
#include "IEUtility/IEMath.h"
#include "CudaInit.h"
#include "CVoxel.cuh"

__global__ void PurgePages(CVoxelPage* gVoxelData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;

	if(pageLocalId == 0) gVoxelData[pageId].dEmptySegmentStackSize = GI_SEGMENT_PER_PAGE;
	if(pageLocalId < GI_SEGMENT_PER_PAGE)
	{
		SegmentObjData s = {0};
		//s.occupation = SegmentOccupation::EMPTY;

		gVoxelData[pageId].dSegmentObjData[pageLocalId] = s;
		gVoxelData[pageId].dEmptySegmentPos[pageLocalId] = GI_SEGMENT_PER_PAGE - globalId - 1;
	}
	gVoxelData[pageId].dGridVoxNorm[pageLocalId] = 0xFFFFFFFF;
	gVoxelData[pageId].dGridVoxPos[pageLocalId] = 0xFFFFFFFF;
}

// Small Helper Kernel That used to init inital obj Pages
// Logic is per segment
__global__ void EmptyPageInit(unsigned char* gPageEmptySegmentPos)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= GI_SEGMENT_PER_PAGE) return;
	gPageEmptySegmentPos[globalId] = GI_SEGMENT_PER_PAGE - globalId - 1;
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
__global__ void DetermineTotalSegment(int& dTotalSegmentCount,
									  // Per object Related (Write)
									  unsigned int* gObjectVoxStrides,
									  unsigned int* gObjectAllocIndexLookup,	  
									  // Per object Related (Read)									  
									  const CObjectVoxelInfo* gVoxelInfo,
									  const uint32_t objCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;
	
	// Determine Strides
	// Here this implementation is slow and does redundant work 
	// but its most easily written version
	unsigned int objStirde = 0, objIndexLookup = 0;
	for(unsigned int i = 0; i < globalId; i++)
	{	
		objStirde += gVoxelInfo[i].voxelCount;
		objIndexLookup += (gVoxelInfo[i].voxelCount + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
	}

	gObjectVoxStrides[globalId] = objStirde;
	gObjectAllocIndexLookup[globalId] = objIndexLookup;

	if(globalId == objCount - 1)
	{
		// Determine Segment Count and add do total segment counter
		unsigned int segmentCount = (gVoxelInfo[globalId].voxelCount + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
		dTotalSegmentCount = objIndexLookup + segmentCount;
	}
}

// Used to populate segment object id's
// Logic per object in batch
__global__ void DetermineSegmentObjId(SegmentObjData* gSegmentObjectId,

									  // Grid Related
									  const CVoxelGrid& gGridInfo,

									  const unsigned int* gObjectAllocIndexLookup,
									  const unsigned int* gObjectVoxStrides,
									  const CObjectVoxelInfo* gVoxelInfo,
									  const CObjectTransform* gObjTransforms,
									  const CVoxelIds* gVoxIds,
									  const uint32_t* gObjTransformIds,

									  const uint32_t batchId,
									  const uint32_t objCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;

	uint32_t transformId = gObjTransformIds[globalId];
	float3 scaling = ExtractScaleInfo(gObjTransforms[transformId].transform);
	assert(fabs(scaling.x - scaling.y) < 0.1f);
	assert(fabs(scaling.y - scaling.z) < 0.1f);
	unsigned int voxelDim = static_cast<unsigned int>((gVoxelInfo[globalId].span * scaling.x + 0.1f) / gGridInfo.span);
	unsigned int segmentCount = (gVoxelInfo[globalId].voxelCount + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE;
	unsigned int voxStart = (voxelDim == 0) ? 0xFFFFFFFF : globalId;

	// Get Vox Id For Obj Type
	unsigned int voxStride = gObjectVoxStrides[globalId];
	CVoxelIds voxId = gVoxIds[voxStride];

	// Fetch Id and obj type
	CVoxelObjectType objType;
	ushort2 objectId;
	unsigned int renderLoc;
	ExpandVoxelIds(renderLoc, objectId, objType, voxId);

	SegmentObjData s;
	s.batchId = static_cast<uint16_t>(batchId);
	s.objectSegmentId = gObjectAllocIndexLookup[globalId];
	s.packed = PackSegmentPacked(objType, SegmentOccupation::OCCUPIED, 1024);
	s.objId = static_cast<uint16_t>(voxStart);
	s.voxStride = voxStride;

	for(unsigned int i = 0; i < segmentCount; i++)
	{
		if(i == segmentCount - 1)
		{
			// Last segment May not have fully occupied
			uint16_t segmentUsage = gVoxelInfo[globalId].voxelCount % GI_SEGMENT_SIZE;
			segmentUsage = (segmentUsage == 0) ? 1024 : segmentUsage;
			s.packed = PackSegmentPacked(objType, SegmentOccupation::OCCUPIED, segmentUsage);
		}
		gSegmentObjectId[gObjectAllocIndexLookup[globalId] + i] = s;
		s.objectSegmentId++;
		s.voxStride += GI_SEGMENT_SIZE;
	}
}

const unsigned int GICudaAllocator::SVOTextureSize = 64;
const unsigned int GICudaAllocator::SVOTextureDepth = 7;

GICudaAllocator::GICudaAllocator(const CVoxelGrid& gridInfo)
	: totalObjectCount(0)
	, totalSegmentCount(0)
	, dVoxelGridInfo(1)
	, hVoxelGridInfo(gridInfo)
	, pointersSet(false)
{
	dVoxelGridInfo.Assign(0, hVoxelGridInfo);
}

void GICudaAllocator::LinkOGLVoxelCache(GLuint aabbBuffer,
										GLuint transformBuffer,
										GLuint jointTransformBuffer,
										GLuint transformIDBuffer,
										GLuint infoBuffer,
										GLuint voxelNormPosBuffer,
										GLuint voxelIdsBuffer,
										GLuint voxelCacheRender,
										GLuint voxelWeightBuffer,
										uint32_t objCount,
										uint32_t voxelCount)
{
	CudaTimer timer(0);
	timer.Start();

	if(voxelCount == 0) return;

	transformLinks.emplace_back();
	transformIdLinks.emplace_back();
	aabbLinks.emplace_back();
	objectInfoLinks.emplace_back();
	cacheNormPosLinks.emplace_back();
	cacheRenderLinks.emplace_back();
	jointTransformLinks.emplace_back();
	objWeightLinks.emplace_back();

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&transformLinks.back(), transformBuffer, cudaGraphicsMapFlagsReadOnly));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&transformIdLinks.back(), transformIDBuffer, cudaGraphicsMapFlagsReadOnly));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&aabbLinks.back(), aabbBuffer, cudaGraphicsMapFlagsReadOnly));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&objectInfoLinks.back(), infoBuffer, cudaGraphicsMapFlagsReadOnly));

	if(jointTransformBuffer != 0)
	{
		CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&jointTransformLinks.back(), jointTransformBuffer, cudaGraphicsMapFlagsReadOnly));
		CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&objWeightLinks.back(), voxelWeightBuffer, cudaGraphicsMapFlagsReadOnly));
	}

	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cacheNormPosLinks.back(), voxelNormPosBuffer, cudaGraphicsMapFlagsReadOnly));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cacheRenderLinks.back(), voxelCacheRender, cudaGraphicsMapFlagsReadOnly));

	objectCounts.emplace_back(objCount);
	voxelCounts.emplace_back(voxelCount);
	totalObjectCount += objCount;

	// Allocate Helper Data
	dVoxelStrides.emplace_back(objCount);
	dObjectAllocationIndexLookup.emplace_back(objCount);

	dVoxelStrides.back().Memset(0, 0, dVoxelStrides.back().Size());
	dObjectAllocationIndexLookup.back().Memset(0, 0, dObjectAllocationIndexLookup.back().Size());
	
	// Populate Helper Data
	// Determine object segement sizes
	int* dTotalCount = nullptr;
	int hTotalCount = 0;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTotalCount), sizeof(int)));
	CUDA_CHECK(cudaMemcpy(dTotalCount, &hTotalCount, sizeof(int), cudaMemcpyHostToDevice));

	// Temporary Linking ObjId Buffer
	cudaGraphicsResource_t tempIdBuffer;
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&tempIdBuffer, voxelIdsBuffer, cudaGraphicsMapFlagsReadOnly));

	// Mapping Pointer
	CVoxelIds* dVoxelIds = nullptr;
	CObjectVoxelInfo* dVoxelInfo = nullptr;
	CObjectTransform* dObjTransform = nullptr;
	uint32_t* dObjTransformIds = nullptr;
	size_t size = 0;
	CUDA_CHECK(cudaGraphicsMapResources(1, &objectInfoLinks.back()));
	CUDA_CHECK(cudaGraphicsMapResources(1, &transformLinks.back()));
	CUDA_CHECK(cudaGraphicsMapResources(1, &transformIdLinks.back()));
	CUDA_CHECK(cudaGraphicsMapResources(1, &tempIdBuffer));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxelInfo), &size, objectInfoLinks.back()));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dObjTransform), &size, transformLinks.back()));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dObjTransformIds), &size, transformIdLinks.back()));
	CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&dVoxelIds), &size, tempIdBuffer));
	
	uint32_t gridSize = static_cast<uint32_t>((objCount + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK);
	DetermineTotalSegment<<<gridSize, GI_THREAD_PER_BLOCK>>>(*dTotalCount,
															 dVoxelStrides.back().Data(),
															 dObjectAllocationIndexLookup.back().Data(),

															 dVoxelInfo,
															 objCount);
	CUDA_KERNEL_CHECK();
	
	// Allocation after determining total index count
	CUDA_CHECK(cudaMemcpy(&hTotalCount, dTotalCount, sizeof(int), cudaMemcpyDeviceToHost));
	dSegmentObjecId.emplace_back(hTotalCount);
	dSegmentAllocLoc.emplace_back(hTotalCount);
	dSegmentAllocLoc.back().Memset(0xFF, 0, dSegmentAllocLoc.back().Size());
	totalSegmentCount += hTotalCount;

	gridSize = static_cast<uint32_t>((objCount + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK);
	DetermineSegmentObjId<<<gridSize, GI_THREAD_PER_BLOCK>>>(dSegmentObjecId.back().Data(),
															 *dVoxelGridInfo.Data(),

															 dObjectAllocationIndexLookup.back().Data(),
															 dVoxelStrides.back().Data(),
															 dVoxelInfo,
															 dObjTransform,
															 dVoxelIds,
															 dObjTransformIds,

															 static_cast<uint32_t>(transformLinks.size() - 1),
															 objCount);
	CUDA_KERNEL_CHECK();


	////DEBUG
	//dObjectAllocationIndexLookup.back().DumpToFile("allocIndexLookup");
	//dVoxelStrides.back().DumpToFile("voxelStrides");
	//dSegmentObjecId.back().DumpToFile("segmentObjId");
	//dSegmentAllocLoc.back().DumpToFile("segmentAllocLoc");

	//std::vector<CObjectVoxelInfo> objInfoArray;
	//objInfoArray.resize(objCount);
	//cudaMemcpy(objInfoArray.data(), dVoxelInfo, objCount * sizeof(CObjectVoxelInfo), cudaMemcpyDeviceToHost);

	//std::ofstream fOut;
	//fOut.open("objVoxelInfo");

	//for(const CObjectVoxelInfo& data : objInfoArray)
	//{
	//	fOut << "{" << data.span << ", " << data.voxelCount << "}" << std::endl;
	//}
	//fOut.close();
	////DEBUG END

	CUDA_CHECK(cudaGraphicsUnmapResources(1, &objectInfoLinks.back()));
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &transformLinks.back()));
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &transformIdLinks.back()));
	CUDA_CHECK(cudaGraphicsUnmapResources(1, &tempIdBuffer));

	dObjectAllocationIndexLookup2D.InsertEnd(dObjectAllocationIndexLookup.back().Data());
	dSegmentAllocLoc2D.InsertEnd(dSegmentAllocLoc.back().Data());
	dObjectVoxStrides2D.InsertEnd(dVoxelStrides.back().Data());
	CUDA_CHECK(cudaFree(dTotalCount));
	timer.Stop();
	GI_LOG("Linked Object Batch to CUDA. Elaped time %f ms", timer.ElapsedMilliS());

	assert(transformLinks.size() == transformIdLinks.size());
	assert(transformIdLinks.size() == aabbLinks.size());
	assert(aabbLinks.size() == jointTransformLinks.size());
	assert(jointTransformLinks.size() == objectInfoLinks.size());
	assert(objectInfoLinks.size() == cacheNormPosLinks.size());
	assert(cacheNormPosLinks.size() == cacheRenderLinks.size());
	assert(cacheRenderLinks.size() == objWeightLinks.size());
	assert(objWeightLinks.size() == dVoxelStrides.size());
	assert(dVoxelStrides.size() == dObjectAllocationIndexLookup.size());
	assert(dObjectAllocationIndexLookup.size() == dSegmentObjecId.size());
	assert(dSegmentObjecId.size() == dSegmentAllocLoc.size());
}

void GICudaAllocator::SetupDevicePointers()
{
	assert(pointersSet == false);

	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(transformLinks.size()), transformLinks.data()));
	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(transformIdLinks.size()), transformIdLinks.data()));
	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data()));
	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data()));

	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(cacheNormPosLinks.size()), cacheNormPosLinks.data()));
	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data()));

	size_t size = 0;
	for(unsigned int i = 0; i < objectCounts.size(); i++)
	{
		hTransforms.push_back(nullptr);
		hTransformIds.push_back(nullptr);
		hObjectAABB.push_back(nullptr);
		hObjectInfo.push_back(nullptr);
		hJointTransform.push_back(nullptr);
		
		hObjNormPosCache.push_back(nullptr);
		hObjRenderCache.push_back(nullptr);
		hObjWeight.push_back(nullptr);

		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hTransforms.back()), &size, transformLinks[i]));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hTransformIds.back()), &size, transformIdLinks[i]));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjectAABB.back()), &size, aabbLinks[i]));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjectInfo.back()), &size, objectInfoLinks[i]));

		if(jointTransformLinks[i] != nullptr)
		{
			CUDA_CHECK(cudaGraphicsMapResources(1, &jointTransformLinks[i]));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hJointTransform.back()), &size, jointTransformLinks[i]));

			CUDA_CHECK(cudaGraphicsMapResources(1, &objWeightLinks[i]));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjWeight.back()), &size, objWeightLinks[i]));
		}

		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjNormPosCache.back()), &size, cacheNormPosLinks[i]));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&hObjRenderCache.back()), &size, cacheRenderLinks[i]));
	}

	// Data Sent to GPU
	dTransforms = hTransforms;
	dTransformIds = hTransformIds;
	dObjectAABB = hObjectAABB;
	dObjectInfo = hObjectInfo;
	dJointTransform = hJointTransform;

	dObjNormPosCache = hObjNormPosCache;
	dObjRenderCache = hObjRenderCache;
	dObjWeight = hObjWeight;
	pointersSet = true;
}

void GICudaAllocator::ClearDevicePointers()
{
	assert(pointersSet == true);
	dTransforms.Clear();
	dTransformIds.Clear();
	dObjectAABB.Clear();
	dObjectInfo.Clear();
	dJointTransform.Clear();

	dObjNormPosCache.Clear();
	dObjRenderCache.Clear();

	hTransforms.clear();
	hTransformIds.clear();
	hObjectAABB.clear();
	hObjectInfo.clear();
	hJointTransform.clear();

	hObjNormPosCache.clear();
	hObjRenderCache.clear();
	
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(transformLinks.size()), transformLinks.data()));
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(transformIdLinks.size()), transformIdLinks.data()));
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(aabbLinks.size()), aabbLinks.data()));
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(objectInfoLinks.size()), objectInfoLinks.data()));

	for(int i = 0; i < jointTransformLinks.size(); i++)
	{
		if(jointTransformLinks[i] != nullptr)
		{
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &jointTransformLinks[i]));
			CUDA_CHECK(cudaGraphicsUnmapResources(1, &objWeightLinks[i]));
		}
	}

	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(cacheNormPosLinks.size()), cacheNormPosLinks.data()));
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(cacheRenderLinks.size()), cacheRenderLinks.data()));
	pointersSet = false;
}

void GICudaAllocator::AddVoxelPages(size_t count)
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
		CUDA_KERNEL_CHECK();
		hPageData.back().dIsSegmentOccupied.Memset(0, 0, hPageData.back().dIsSegmentOccupied.Size());
		hPageData.back().dVoxelPageNorm.Memset(0xFF, 0, hPageData.back().dVoxelPageNorm.Size());
		hPageData.back().dVoxelOccupancy.Memset(0xFF, 0, hPageData.back().dVoxelOccupancy.Size());
		hPageData.back().dVoxelPagePos.Memset(0xFF, 0, hPageData.back().dVoxelPagePos.Size());
		
		CVoxelPage voxData =
		{
			hPageData.back().dVoxelPagePos.Data(),
			hPageData.back().dVoxelPageNorm.Data(),
			hPageData.back().dVoxelOccupancy.Data(),
			hPageData.back().dEmptySegmentList.Data(),
			hPageData.back().dIsSegmentOccupied.Data(),
			GI_SEGMENT_PER_PAGE
		};
		hVoxelPages.push_back(voxData);
	}
	dVoxelPages = hVoxelPages;

	////DEBUG
	//hPageData.front().dEmptySegmentList.DumpToFile("emptySegmentListFirstPage");
	//hPageData.front().dIsSegmentOccupied.DumpToFile("isSegmentOcuupListFirstPage");
}

void GICudaAllocator::RemoveVoxelPages(size_t count)
{
	hPageData.resize(hPageData.size() - count);
	hVoxelPages.resize(hVoxelPages.size() - count);
	dVoxelPages = hVoxelPages;
}

void GICudaAllocator::ResetSceneData()
{
	for(unsigned int i = 0; i < transformLinks.size(); i++)
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(transformLinks[i]));
		CUDA_CHECK(cudaGraphicsUnregisterResource(transformIdLinks[i]));
		CUDA_CHECK(cudaGraphicsUnregisterResource(aabbLinks[i]));
		CUDA_CHECK(cudaGraphicsUnregisterResource(objectInfoLinks[i]));

		if(jointTransformLinks[i] != nullptr)
		{
			CUDA_CHECK(cudaGraphicsUnregisterResource(jointTransformLinks[i]));
			CUDA_CHECK(cudaGraphicsUnregisterResource(objWeightLinks[i]));
		}

		CUDA_CHECK(cudaGraphicsUnregisterResource(cacheNormPosLinks[i]));
		CUDA_CHECK(cudaGraphicsUnregisterResource(cacheRenderLinks[i]));
	}	

	transformLinks.clear();
	transformIdLinks.clear();
	aabbLinks.clear();
	objectInfoLinks.clear();
	jointTransformLinks.clear();

	cacheNormPosLinks.clear();
	cacheRenderLinks.clear();
	objWeightLinks.clear();

	dSegmentObjecId.clear();
	dSegmentAllocLoc.clear();

	dVoxelStrides.clear();
	dObjectAllocationIndexLookup.clear();
	
	dObjectAllocationIndexLookup2D.Clear();
	dSegmentAllocLoc2D.Clear();

	objectCounts.clear();
	voxelCounts.clear();

	totalObjectCount = 0;
	totalSegmentCount = 0;

	// Clear all voxel pages (normally resetting values should suffice but there is a bug)
	// And couldnt figure out where it is
	dVoxelPages.Clear();
	hVoxelPages.clear();
	hPageData.clear();
	reservedPageCount = 0;

	//// Reset Voxel Values
	//if(dVoxelPages.Size() > 0)
	//{
	//	unsigned int gridSize = (dVoxelPages.Size() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
	//							 GI_THREAD_PER_BLOCK;
	//	PurgePages<<<gridSize, GI_THREAD_PER_BLOCK>>>(GetVoxelPagesDevice());
	//	CUDA_KERNEL_CHECK();
	//}
}

void GICudaAllocator::ReserveForSegments(float coverageRatio)
{
	uint32_t requiredPages = static_cast<uint32_t>((totalSegmentCount + GI_SEGMENT_PER_PAGE - 1) / GI_SEGMENT_PER_PAGE);
	requiredPages = static_cast<uint32_t>(requiredPages * coverageRatio);

	if(dVoxelPages.Size() < requiredPages)
	{
		AddVoxelPages(requiredPages - dVoxelPages.Size());
	}
	else if(dVoxelPages.Size() > requiredPages)
	{
		RemoveVoxelPages(dVoxelPages.Size() - requiredPages);
	}
	reservedPageCount = requiredPages;
}

uint64_t GICudaAllocator::SystemTotalMemoryUsage() const
{
	uint64_t memory = 0;

	// Page Pointer Lookup Array
	memory += dVoxelPages.Size() * sizeof(CVoxelPage);

	// Actual Page Data
	memory += dVoxelPages.Size() * GI_SEGMENT_PER_PAGE * (sizeof(SegmentObjData) +
														  sizeof(unsigned char));

	memory += dVoxelPages.Size() * GI_PAGE_SIZE * (sizeof(CVoxelNormPos) + sizeof(CVoxelOccupancy));

	// Object Batch Helpers
	for(unsigned int i = 0; i < dSegmentObjecId.size(); i++)
	{
		// Per Segment Related
		memory += dSegmentObjecId[i].Size() * (sizeof(SegmentObjData) +
											   sizeof(ushort2));
		
		// Per Object Related
		memory += dVoxelStrides.size() * (sizeof(unsigned int) + 
										  sizeof(unsigned int) + 
										  sizeof(unsigned int));
	}

	memory += dObjectAllocationIndexLookup2D.Size() * (sizeof(unsigned int*) +
													   sizeof(unsigned int*) +
													   sizeof(ushort2*));
	return memory;
}

void GICudaAllocator::SendNewVoxPosToDevice()
{
	dVoxelGridInfo.Assign(0, hVoxelGridInfo);
}

uint32_t GICudaAllocator::NumObjectBatches() const
{
	return static_cast<uint32_t>(transformLinks.size());
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
	return static_cast<uint32_t>(reservedPageCount);
}

uint32_t GICudaAllocator::NumSegments() const
{
	return static_cast<uint32_t>(totalSegmentCount);
}

CVoxelGrid* GICudaAllocator::GetVoxelGridDevice()
{
	return dVoxelGridInfo.Data();
}

const CVoxelGrid& GICudaAllocator::GetVoxelGridHost() const
{
	return hVoxelGridInfo;
}

IEVector3 GICudaAllocator::GetNewVoxelPos(const IEVector3& playerPos, float cascadeMultiplier)
{
	// only grid span increments are allowed
	float rootSpan = hVoxelGridInfo.span * cascadeMultiplier;
	//float rootSnapLevelMultiplier = 64.0f;
	float rootSnapLevelMultiplier = 16.0f;
    //float rootSnapLevelMultiplier = 4.0f;
	//float rootSnapLevelMultiplier = 1.0f;

	float3 voxelCornerPos;
	voxelCornerPos.x = playerPos.getX() - rootSpan * hVoxelGridInfo.dimension.x * 0.5f;
	voxelCornerPos.y = playerPos.getY() - rootSpan * hVoxelGridInfo.dimension.y * 0.5f;
	voxelCornerPos.z = playerPos.getZ() - rootSpan * hVoxelGridInfo.dimension.z * 0.5f;
	
	rootSpan *= rootSnapLevelMultiplier;
	voxelCornerPos.x -= std::fmod(voxelCornerPos.x + rootSpan * 0.5f, rootSpan);
	voxelCornerPos.y -= std::fmod(voxelCornerPos.y + rootSpan * 0.5f, rootSpan);
	voxelCornerPos.z -= std::fmod(voxelCornerPos.z + rootSpan * 0.5f, rootSpan);

	// VoxCornerPos is at root cascade's position
	// Translate it to cascade corner
	if(cascadeMultiplier != 0)
	{
		float cascadeNo = std::log2(cascadeMultiplier);
		float factor = IEMathFunctions::SumLinear(cascadeNo);
		voxelCornerPos.x += (factor * hVoxelGridInfo.span * hVoxelGridInfo.dimension.x) / 2.0f;
		voxelCornerPos.y += (factor * hVoxelGridInfo.span * hVoxelGridInfo.dimension.y) / 2.0f;
		voxelCornerPos.z += (factor * hVoxelGridInfo.span * hVoxelGridInfo.dimension.z) / 2.0f;
	}
	hVoxelGridInfo.position.x = voxelCornerPos.x;
	hVoxelGridInfo.position.y = voxelCornerPos.y;
	hVoxelGridInfo.position.z = voxelCornerPos.z;
	return 
	{
		hVoxelGridInfo.position.x, 
		hVoxelGridInfo.position.y, 
		hVoxelGridInfo.position.z
	};
}

CObjectTransform** GICudaAllocator::GetTransformsDevice()
{
	return dTransforms.Data();
}

uint32_t** GICudaAllocator::GetTransformIDDevice()
{
	return dTransformIds.Data();
}

CObjectAABB** GICudaAllocator::GetObjectAABBDevice()
{
	return dObjectAABB.Data();
}

CObjectVoxelInfo** GICudaAllocator::GetObjectInfoDevice()
{
	return dObjectInfo.Data();
}

CObjectTransform** GICudaAllocator::GetJointTransformDevice()
{
	return dJointTransform.Data();
}

CVoxelNormPos** GICudaAllocator::GetObjCacheNormPosDevice()
{
	return dObjNormPosCache.Data();
}

CVoxelColor** GICudaAllocator::GetObjRenderCacheDevice()
{
	return dObjRenderCache.Data();
}

CVoxelWeight** GICudaAllocator::GetObjWeightDevice()
{
	return dObjWeight.Data();
}

CObjectTransform* GICudaAllocator::GetTransformsDevice(uint32_t index)
{
	return hTransforms[index];
}

uint32_t* GICudaAllocator::GetTransformIDDevice(uint32_t index)
{
	return hTransformIds[index];
}

CObjectAABB* GICudaAllocator::GetObjectAABBDevice(uint32_t index)
{
	return hObjectAABB[index];
}

CObjectVoxelInfo* GICudaAllocator::GetObjectInfoDevice(uint32_t index)
{
	return hObjectInfo[index];
}

CVoxelNormPos* GICudaAllocator::GetObjCacheNormPosDevice(uint32_t index)
{
	return hObjNormPosCache[index];
}

CVoxelColor* GICudaAllocator::GetObjRenderCacheDevice(uint32_t index)
{
	return hObjRenderCache[index];
}

CVoxelPage* GICudaAllocator::GetVoxelPagesDevice()
{
	return dVoxelPages.Data();
}

SegmentObjData* GICudaAllocator::GetSegmentObjectID(uint32_t index)
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

bool GICudaAllocator::IsGLMapped()
{
	return pointersSet;
}