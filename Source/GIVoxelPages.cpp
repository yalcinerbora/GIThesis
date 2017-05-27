#include "GIVoxelPages.h"
#include "PageKernels.cuh"

static std::ostream& operator<<(std::ostream& ostr, const uint2& int2)
{
	ostr << "{" << int2.x << ", " << int2.y << "}";
	return ostr;
}

static std::ostream& operator<<(std::ostream& ostr, const SegmentOccupation& seg)
{
	ostr << static_cast<int>(seg);
	return ostr;
}

static std::ostream& operator<<(std::ostream& ostr, const SegmentObjData& segObj)
{
	uint16_t objType = segObj.packed >> 14;
	uint16_t occupation = (segObj.packed >> 11) & 0x000F;
	uint16_t segmentO = segObj.packed & 0x07FF;

	ostr << segObj.batchId << " ";
	ostr << segObj.objId << " | ";
	ostr << segObj.objectSegmentId << " | ";
	ostr << objType << " ";
	ostr << occupation << " ";
	ostr << segmentO << " ";
	ostr << segObj.voxStride;

	return ostr;
}

VoxelPageData::VoxelPageData(size_t pageCount)
{
	assert(pageCount != 0);
	size_t sizePerPage = GIVoxelPages::PageSize *
						 (sizeof(CVoxelPos) +
						  sizeof(CVoxelNorm) +
						  sizeof(CVoxelOccupancy))
						 +
						 GIVoxelPages::SegmentSize *
						 (sizeof(unsigned char) +
						  sizeof(SegmentObjData));

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

		page.dSegmentObjData = reinterpret_cast<SegmentObjData*>(dPtr + offset);
		offset += GIVoxelPages::SegmentPerPage * sizeof(SegmentObjData);

		page.dEmptySegmentStackSize = GIVoxelPages::SegmentPerPage;
		pages.push_back(page);
	}
	assert(offset == pageData.Size());

	// KC to Initialize Empty Segment Stack
	int blockSize = CudaInit::GenBlockSizeSmall(pageCount * GIVoxelPages::SegmentPerPage);
	InitializePage<<<blockSize, CudaInit::TPB>>>(pages.front().dEmptySegmentPos,
												 sizePerPage, pageCount);
}

VoxelPageData::VoxelPageData(VoxelPageData&& other)
	: pageData(std::move(other.pageData))
	, pages(std::move(other.pages))
{}

size_t VoxelPageData::PageCount() const
{
	return pages.size();
}

const std::vector<CVoxelPage>& VoxelPageData::Pages() const
{
	return pages;
}


//----------------------//

GIVoxelPages(SceneI& scene);
GIVoxelPages(const GIVoxelPages&) = delete;
GIVoxelPages&				operator=(const GIVoxelPages&) = delete;
GIVoxelPages(GIVoxelPages&&);
~GIVoxelPages() = default;