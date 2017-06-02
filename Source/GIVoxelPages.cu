#include "GIVoxelPages.h"
#include "PageKernels.cuh"
#include "DrawBuffer.h"
#include "CudaInit.h"
#include "CudaTimer.h"
#include "GIVoxelCache.h"
#include "GISparseVoxelOctree.h"
#include "MeshBatchSkeletal.h"
#include "IEUtility/IEMath.h"
#include <cuda_gl_interop.h>
inline static std::ostream& operator<<(std::ostream& ostr, const uint2& int2)
{
	ostr << "{" << int2.x << ", " << int2.y << "}";
	return ostr;
}

inline static std::ostream& operator<<(std::ostream& ostr, const CSegmentOccupation& seg)
{
	ostr << static_cast<int>(seg);
	return ostr;
}

inline static std::ostream& operator<<(std::ostream& ostr, const CSegmentInfo& segObj)
{
	uint16_t cascadeNo = (segObj.packed >> 14) & 0x0003;
	uint16_t objType = (segObj.packed >> 12) & 0x0003;
	uint16_t occupation = (segObj.packed >> 10) & 0x0003;

	ostr << cascadeNo << ", ";
	ostr << segObj.batchId << ", ";
	ostr << segObj.objId << " | ";

	ostr << segObj.objectSegmentId << " | ";
	ostr << objType << " | ";
	ostr << occupation << " | ";
	return ostr;
}

GIVoxelPages::MultiPage::MultiPage(size_t pageCount)
{
	assert(pageCount != 0);
	size_t sizePerPage = GIVoxelPages::PageSize *
						 (sizeof(CVoxelPos) +
						  sizeof(CVoxelNorm) +
						  sizeof(CVoxelOccupancy))
						 +
						 GIVoxelPages::SegmentSize *
						 (sizeof(unsigned char) +
						  sizeof(CSegmentInfo));

	size_t totalSize = sizePerPage * pageCount;
	pageData.Resize(totalSize);
	pageData.Memset(0x0, 0, totalSize);

	uint8_t* dPtr = pageData.Data();
	ptrdiff_t offset = 0;
	for(size_t i = 0; i < pageCount; i++)
	{
		CVoxelPage page = {};

		page.dGridVoxPos = reinterpret_cast<CVoxelPos*>(dPtr + offset);
		offset += GIVoxelPages::PageSize * sizeof(CVoxelPos);

		page.dGridVoxNorm = reinterpret_cast<CVoxelNorm*>(dPtr + offset);
		offset += GIVoxelPages::PageSize * sizeof(CVoxelNorm);

		page.dGridVoxOccupancy = reinterpret_cast<CVoxelOccupancy*>(dPtr + offset);
		offset += GIVoxelPages::PageSize * sizeof(CVoxelNorm);

		page.dEmptySegmentPos = reinterpret_cast<unsigned char*>(dPtr + offset);
		offset += GIVoxelPages::SegmentPerPage * sizeof(unsigned char);

		page.dSegmentInfo = reinterpret_cast<CSegmentInfo*>(dPtr + offset);
		offset += GIVoxelPages::SegmentPerPage * sizeof(CSegmentInfo);

		page.dEmptySegmentStackSize = GIVoxelPages::SegmentPerPage;
		pages.push_back(page);
	}
	assert(offset == pageData.Size());

	// KC to Initialize Empty Segment Stack
	int blockSize = CudaInit::GenBlockSizeSmall(static_cast<uint32_t>(pageCount * GIVoxelPages::SegmentPerPage));
	int tbb = CudaInit::TBP;
	InitializePage<<<blockSize, tbb>>>(pages.front().dEmptySegmentPos, pageCount);
}

GIVoxelPages::MultiPage::MultiPage(MultiPage&& other)
	: pageData(std::move(other.pageData))
	, pages(std::move(other.pages))
{}

size_t GIVoxelPages::MultiPage::PageCount() const
{
	return pages.size();
}

const std::vector<CVoxelPage>& GIVoxelPages::MultiPage::Pages() const
{
	return pages;
}

uint16_t GIVoxelPages::PackSegmentInfo(const uint8_t cascadeId,
									   const CObjectType type,
									   const CSegmentOccupation occupation)
{
	// MSB to LSB 
	// 2 bit cascadeId
	// 2 bit object type 
	// 2 bit segment occupation
	uint16_t packed = 0;
	packed |= (static_cast<uint16_t>(cascadeId) & 0x0003) << 14;
	packed |= (static_cast<uint16_t>(type) & 0x0003) << 12;
	packed |= (static_cast<uint16_t>(occupation) & 0x0003) << 10;
	return packed;
}

void GIVoxelPages::GenerateGPUData(const GIVoxelCache& cache)
{
	// Generate SegInfos
	std::vector<CSegmentInfo> segInfos;
	uint32_t mapOGLBufferCount = 0;

	for(uint32_t cascadeId = 0; cascadeId < svoParams->CascadeCount; cascadeId++)
	for(uint32_t batchId = 0; batchId < batches->size(); batchId++)
	{
		bool nonRigid = (*batches)[batchId]->MeshType() == MeshBatchType::SKELETAL;
		const std::vector<CMeshVoxelInfo> voxInfo = cache.CopyMeshObjectInfo(cascadeId, batchId);
		for(uint32_t objId = 0; objId <  voxInfo.size(); objId++)
		{
			const CMeshVoxelInfo& info = voxInfo[objId];

			uint32_t segmentCount = (info.voxCount + SegmentSize - 1) / SegmentSize;
			for(uint32_t segId = 0; segId < segmentCount; segId++)
			{
				
				CObjectType objType = (nonRigid) ? CObjectType::SKEL_DYNAMIC : CObjectType::DYNAMIC;
				
				CSegmentInfo segInfo;
				segInfo.batchId = static_cast<uint16_t>(batchId);
				segInfo.objectSegmentId = static_cast<uint16_t>(segId);
				segInfo.objId = static_cast<uint16_t>(objId);
				segInfo.packed = PackSegmentInfo(static_cast<uint8_t>(cascadeId), objType, 
												 CSegmentOccupation::OCCUPIED);

				segInfos.push_back(segInfo);
			}
		}

		if(cascadeId == 0)
		{
			if(nonRigid)
			{
				mapOGLBufferCount += 2;
			}
			else mapOGLBufferCount += 1;
		}		
	}

	size_t bufferSize = segInfos.size() * (sizeof(CSegmentInfo) +
										   sizeof(ushort2));
	bufferSize += mapOGLBufferCount * sizeof(BatchOGLData);
	bufferSize += svoParams->CascadeCount * sizeof(CVoxelGrid);

	gpuData.Resize(bufferSize);

	//...

	// Segment Info generated
	// allocate for segment alloc info
	// allocate for batch oglData
	// gen Voxel grids

}

void GIVoxelPages::AllocatePages(size_t voxelCapacity)
{
	size_t pageCount = (voxelCapacity + PageSize - 1) / PageSize;
	size_t oldSize = dPages.Size();

	hPages.emplace_back(pageCount);
	dPages.Resize(oldSize + hPages.back().PageCount());
	dPages.Assign(oldSize, hPages.back().PageCount(), hPages.back().Pages().data());
}

void GIVoxelPages::MapOGLResources()
{
	CUDA_CHECK(cudaGraphicsMapResources(static_cast<int>(batchOGLResources.size()), batchOGLResources.data()));

	std::vector<BatchOGLData> newOGLData;
	size_t batchIndex = 0;
	for(size_t i = 0; i < batches->size(); i++)
	{
		size_t size;
		uint8_t* glPointer = nullptr;
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glPointer),
														&size, batchOGLResources[batchIndex]));

		size_t aabbByteOffset = (*batches)[i]->getDrawBuffer().getAABBOffset();
		size_t modelTransformByteOffset = (*batches)[i]->getDrawBuffer().getModelTransformOffset();
		size_t modelTransformIndexByteOffset = (*batches)[i]->getDrawBuffer().getModelTransformIndexOffset();

		BatchOGLData batchGL = {};
		batchGL.dAABBs = reinterpret_cast<CAABB*>(glPointer + aabbByteOffset);
		batchGL.dModelTransforms = reinterpret_cast<CModelTransform*>(glPointer + modelTransformByteOffset);
		batchGL.dModelTransformIndices = reinterpret_cast<uint32_t*>(glPointer + modelTransformIndexByteOffset);

		if((*batches)[i]->MeshType() == MeshBatchType::SKELETAL)
		{
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&glPointer),
															&size, batchOGLResources[batchIndex]));
			batchGL.dJointTransforms = reinterpret_cast<CJointTransform*>(glPointer);
			batchIndex++;
		}
		batchIndex++;
		newOGLData.push_back(batchGL);
	}

	// Copy generated pointers to GPU
	CUDA_CHECK(cudaMemcpy(dBatchOGLData,
						  newOGLData.data(),
						  batches->size() * sizeof(BatchOGLData),
						  cudaMemcpyHostToDevice));
}

void GIVoxelPages::UnmapOGLResources()
{
	CUDA_CHECK(cudaGraphicsUnmapResources(static_cast<int>(batchOGLResources.size()), batchOGLResources.data()));
}

GIVoxelPages::GIVoxelPages()
	: batches(nullptr)
	, svoParams(nullptr)
	, segmentAmount(0)
	, dVoxelGrids(nullptr)
	, dBatchOGLData(nullptr)
	, dSegmentInfo(nullptr)
	, dSegmentAllocInfo(nullptr)
{}

GIVoxelPages::GIVoxelPages(const GIVoxelCache& cache, 
						   const std::vector<MeshBatchI*>* batches,
						   const OctreeParameters& octreeParams,
						   size_t initalVoxelCapacity)
	: batches(batches)
	, svoParams(svoParams)
	, segmentAmount(0)
	, dVoxelGrids(nullptr)
	, dBatchOGLData(nullptr)
	, dSegmentInfo(nullptr)
	, dSegmentAllocInfo(nullptr)
	, vRenderWorldVoxel(ShaderType::VERTEX, "Shaders/VoxRenderWorld.vert")
	, fRenderWorldVoxel(ShaderType::FRAGMENT, "Shaders/VoxRender.frag")
{
	AllocatePages(initalVoxelCapacity);

	for(uint32_t i = 0; i < batches->size(); i++)
	{
		MeshBatchI& batch = *(*batches)[i];
		GLuint bufferId = batch.getDrawBuffer().getGLBuffer();
		cudaGraphicsResource_t glResource;
		CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&glResource, 
												bufferId,
												cudaGraphicsMapFlagsReadOnly));
		batchOGLResources.push_back(glResource);

		if(batch.MeshType() == MeshBatchType::SKELETAL)
		{
			GLuint jointBuffer = static_cast<MeshBatchSkeletal&>(batch).getJointTransforms().getGLBuffer();
			CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&glResource,
													jointBuffer,
													cudaGraphicsMapFlagsReadOnly));
			batchOGLResources.push_back(glResource);
		}
	}
	GenerateGPUData(cache);
}

GIVoxelPages::GIVoxelPages(GIVoxelPages&& other)
	: batches(other.batches)
	, svoParams(other.svoParams)
	, gpuData(std::move(other.gpuData))
	, dVoxelGrids(other.dVoxelGrids)
	, dBatchOGLData(other.dBatchOGLData)
	, dSegmentInfo(other.dSegmentInfo)
	, dSegmentAllocInfo(other.dSegmentAllocInfo)
	, hPages(std::move(other.hPages))
	, dPages(std::move(other.dPages))
	, batchOGLResources(std::move(other.batchOGLResources))
	, debugDrawBuffer(std::move(other.debugDrawBuffer))
	, debugDrawVao(std::move(other.debugDrawVao))
	, vRenderWorldVoxel(std::move(other.vRenderWorldVoxel))
	, fRenderWorldVoxel(std::move(other.fRenderWorldVoxel))
{
	assert(other.batchOGLResources.empty());
}

GIVoxelPages& GIVoxelPages::operator=(GIVoxelPages&& other)
{
	batches = other.batches;
	svoParams = other.svoParams;
	gpuData = std::move(other.gpuData);
	dVoxelGrids = other.dVoxelGrids;
	dBatchOGLData = other.dBatchOGLData;
	dSegmentInfo = other.dSegmentInfo;
	dSegmentAllocInfo = other.dSegmentAllocInfo;
	hPages = std::move(other.hPages);
	dPages = std::move(other.dPages);
	batchOGLResources = std::move(other.batchOGLResources);
	debugDrawBuffer = std::move(other.debugDrawBuffer);
	debugDrawVao = std::move(other.debugDrawVao);
	vRenderWorldVoxel = std::move(other.vRenderWorldVoxel);
	fRenderWorldVoxel = std::move(other.fRenderWorldVoxel);
	return *this;
}

GIVoxelPages::~GIVoxelPages()
{
	for(cudaGraphicsResource_t resc : batchOGLResources)
	{
		CUDA_CHECK(cudaGraphicsUnregisterResource(resc));
	}
}

void GIVoxelPages::UpdateGridPositions(const IEVector3& cameraPos)
{
	std::vector<IEVector3> positions(svoParams->CascadeCount);
	for(uint32_t i = 0; i < svoParams->CascadeCount; i++)
	{
		// only grid span increments are allowed
		float span = svoParams->BaseSpan * static_cast<float>(1 << i);

		// TODO: Better solution for higher level voxel jittering
		//float rootSnapLevelMultiplier = 64.0f;
		float rootSnapLevelMultiplier = 16.0f;
		//float rootSnapLevelMultiplier = 4.0f;
		//float rootSnapLevelMultiplier = 1.0f;

		IEVector3 voxelCornerPos;
		voxelCornerPos = cameraPos - span * svoParams->CascadeBaseLevelSize * 0.5f;

		span *= rootSnapLevelMultiplier;
		voxelCornerPos[0] -= std::fmod(voxelCornerPos[0] + span * 0.5f, span);
		voxelCornerPos[1] -= std::fmod(voxelCornerPos[1] + span * 0.5f, span);
		voxelCornerPos[2] -= std::fmod(voxelCornerPos[2] + span * 0.5f, span);

		// VoxCornerPos is at root cascade's position
		// Translate it to cascade corner
		if(i != 0)
		{
			float factor = static_cast<float>(IEMathFunctions::SumLinear(i));
			span = svoParams->BaseSpan * static_cast<float>(1 << i);
			voxelCornerPos += (factor * span * svoParams->CascadeBaseLevelSize) / 2.0f;
		}
		positions.push_back(voxelCornerPos);
	}

	// Copy new positions
	CUDA_CHECK(cudaMemcpy2D(dVoxelGrids, sizeof(CVoxelGrid),
							positions.data(), sizeof(IEVector3),
							sizeof(IEVector3), svoParams->CascadeCount,
							cudaMemcpyHostToDevice));
}

double GIVoxelPages::VoxelIO(bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();
	
	// KC
	int blockSize = CudaInit::GenBlockSize(static_cast<int>(segmentAmount));
	int threadSize = CudaInit::TBP;
	VoxelObjectIO<<<blockSize, threadSize>>>(// Voxel System
											 dPages.Data(),
											 dVoxelGrids,
											 // Helper Structures		
											 dSegmentAllocInfo,
											 dSegmentInfo,
											 // Per Object Related
											 dBatchOGLData,
											 // Limits
											 static_cast<uint32_t>(dPages.Size()),
											 static_cast<uint32_t>(dPages.Size()));
	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

double GIVoxelPages::Transform(const GIVoxelCache& cache,
							   bool doTiming)
{
	CudaTimer t;
	if(doTiming) t.Start();

	// KC
	int blockSize = CudaInit::GenBlockSize(static_cast<int>(dPages.Size() * PageSize));
	int threadSize = CudaInit::TBP;
	VoxelTransform<<<blockSize,threadSize>>>(// Voxel Pages
										     dPages.Data(),
										     dVoxelGrids,
										     // OGL Related
										     dBatchOGLData,
										     // Voxel Cache Related
											 cache.getDeviceCascadePointersDevice().Data(),
										     // Limits
										     static_cast<uint32_t>(batches->size()));
	if(doTiming)
	{
		t.Stop();
		return t.ElapsedMilliS();
	}
	return 0.0;
}

uint64_t GIVoxelPages::MemoryUsage() const
{
	size_t totalSize = gpuData.Size();
	totalSize += dPages.Size() * sizeof(CVoxelPage);
	totalSize += dPages.Size() * PageSize * (sizeof(CVoxelPos) +
											 sizeof(CVoxelNorm) +
											 sizeof(CVoxelOccupancy));
	totalSize += dPages.Size() * SegmentPerPage* (sizeof(unsigned char) +
												  sizeof(CSegmentInfo));
	return totalSize;
}

// Debug Draw
void  GIVoxelPages::AllocateDraw()
{

}

void  GIVoxelPages::Draw(size_t cascadeCount)
{

}

void  GIVoxelPages::DeallocateDraw()
{

}

const CVoxelPageConst* GIVoxelPages::getVoxelPages() const
{
	return reinterpret_cast<const CVoxelPageConst*>(dPages.Data());
}

const CVoxelPage* GIVoxelPages::getVoxelPages()
{
	return dPages.Data();
}