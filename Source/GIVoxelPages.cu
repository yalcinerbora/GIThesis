#include "GIVoxelPages.h"
#include "PageKernels.cuh"
#include "DrawBuffer.h"
#include "CudaInit.h"
#include "CudaTimer.h"
#include "GIVoxelCache.h"
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
{

}

GIVoxelPages::GIVoxelPages(const std::vector<MeshBatchI*>* batches,
						   size_t initalVoxelCapacity)
{
	AllocatePages(initalVoxelCapacity);


}

GIVoxelPages::GIVoxelPages(GIVoxelPages&&)
{

}

GIVoxelPages& GIVoxelPages::operator=(GIVoxelPages&&)
{
	return *this;
}

double GIVoxelPages::VoxelIO()
{
	CudaTimer t;
	t.Start();
	
	// KC
	int blockSize = CudaInit::GenBlockSize(static_cast<int>(segmentSize));
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
	t.Stop();
	return t.ElapsedMilliS();
}

double GIVoxelPages::Transform(const GIVoxelCache& cache,
							   const IEVector3 cameraPos)
{
	CudaTimer t;
	t.Start();

	// Determine Grid Positions
	UpdateGridPositions(cameraPos);
	
	// KC
	int blockSize = CudaInit::GenBlockSize(static_cast<int>(dPages.Size() * PageSize));
	int threadSize = CudaInit::TBP;
	VoxelTransform<<<blockSize,threadSize>>>(// Voxel Pages
										     dPages.Data(),
										     dVoxelGrids,
										     dNewGridPositions,
										     // OGL Related
										     dBatchOGLData,
										     // Voxel Cache Related
											 cache.getDeviceCascadePointersDevice().Data(),
										     // Limits
										     static_cast<uint32_t>(batches->size()));
	t.Stop();
	return t.ElapsedMilliS();
}

const CVoxelPageConst* GIVoxelPages::getVoxelPages() const
{
	return reinterpret_cast<const CVoxelPageConst*>(dPages.Data());
}

const CVoxelPage* GIVoxelPages::getVoxelPages()
{
	return dPages.Data();
}