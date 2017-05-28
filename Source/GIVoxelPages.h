#pragma once

// Used in kernel calls that may not saturate enough cores

//#define GI_VOXEL_NEIGBOURS 8

#include <cstdint>
#include <vector>
#include "CudaInit.h"
#include "CudaVector.cuh"
#include "CVoxelTypes.h"

typedef uint2 CVoxelIds;

class SceneI;

// Multi Page Holding Data Structure
class VoxelPageData
{
	private:
		CudaVector<uint8_t>				pageData;
		std::vector<CVoxelPage>			pages;

	protected:

	public:
		// Constructors & Destructor
										VoxelPageData(size_t pageCount = 1);
										VoxelPageData(const VoxelPageData&) = delete;
										VoxelPageData(VoxelPageData&&);
										~VoxelPageData() = default;

		size_t							PageCount() const;
		const std::vector<CVoxelPage>&	Pages() const;
};

class GIVoxelPages
{
	public:
		static constexpr uint32_t	SegmentSize = 1024;
		static constexpr uint32_t	PageSize = 65536;

		static constexpr uint32_t	BlockPerPage = PageSize / CudaInit::TBP;
		static constexpr uint32_t	SegmentPerPage = PageSize / SegmentSize;
		static constexpr uint32_t	SegmentPerBlock = SegmentSize / CudaInit::TBP;

	private:
		//Page System
		std::vector<VoxelPageData>	hPages;
		CudaVector<CVoxelPage>		dPages;



	protected:
	public:
		// Constrcutors & Destructor
									GIVoxelPages(SceneI& scene);
									GIVoxelPages(const GIVoxelPages&) = delete;
		GIVoxelPages&				operator=(const GIVoxelPages&) = delete;
									GIVoxelPages(GIVoxelPages&&);
									~GIVoxelPages() = default;

		// Accessors....
};

static_assert(GIVoxelPages::PageSize % CudaInit::TBP == 0, "Page size must be divisible by thread per block");
static_assert(GIVoxelPages::SegmentPerBlock != 0, "Segment should be bigger(or equal) than block");
static_assert(GIVoxelPages::SegmentSize < 2048, "Segment size should fit on SegmentPacked Structure, requires 10 bits at most");