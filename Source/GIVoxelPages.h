#pragma once

// Used in kernel calls that may not saturate enough cores

//#define GI_VOXEL_NEIGBOURS 8

#include <cstdint>
#include <vector>
#include "CudaInit.h"
#include "CudaVector.cuh"
#include "CVoxelTypes.h"
#include "COpenGLTypes.h"
#include "MeshBatchI.h"

typedef uint2 CVoxelIds;

class SceneI;
class VoxelCache;

struct CModelTransform;

class GIVoxelPages
{
	private:
		// Multi Page Holding Class
		class MultiPage
		{
			private:
				CudaVector<uint8_t>				pageData;
				std::vector<CVoxelPage>			pages;

			protected:

			public:
				// Constructors & Destructor
												MultiPage(size_t pageCount = 1);
												MultiPage(const MultiPage&) = delete;
												MultiPage(MultiPage&&);
												~MultiPage() = default;

				size_t							PageCount() const;
				const std::vector<CVoxelPage>&	Pages() const;
		};

	public:
		static constexpr uint32_t				SegmentSize = 1024;
		static constexpr uint32_t				PageSize = 65536;

		static constexpr uint32_t				BlockPerPage = PageSize / CudaInit::TBP;
		static constexpr uint32_t				SegmentPerPage = PageSize / SegmentSize;
		static constexpr uint32_t				SegmentPerBlock = SegmentSize / CudaInit::TBP;
	
	private:
		// Batch
		const std::vector<MeshBatchI*>*			batches;

		//Page System
		std::vector<MultiPage>					hPages;
		CudaVector<CVoxelPage>					dPages;

		// Helper Data
		CudaVector<CSegmentInfo>				dSegmentInfo;
		CudaVector<ushort2>						dSegmentAllocInfo;

		// OGL Related
		std::vector<cudaGraphicsResource_t>		batchOGLResources;
		CudaVector<BatchOGLData>				dBatchOGLData;

		void									AllocatePages(size_t voxelCapacity);
		void									MapOGLResources();
		void									UnmapOGLResources();

	protected:
	public:
		// Constrcutors & Destructor
												GIVoxelPages();
												GIVoxelPages(const std::vector<MeshBatchI*>* batches,
															 size_t initalVoxelCapacity);
												GIVoxelPages(const GIVoxelPages&) = delete;
												GIVoxelPages(GIVoxelPages&&);
		GIVoxelPages&							operator=(const GIVoxelPages&) = delete;
		GIVoxelPages&							operator=(GIVoxelPages&&);
												~GIVoxelPages() = default;

		double									VoxelIO();
		double									VoxelTransform(VoxelCache& cache);
	
		const CVoxelPageConst*					getVoxelPages() const;
		const CVoxelPage*						getVoxelPages();
};

static_assert(GIVoxelPages::PageSize % CudaInit::TBP == 0, "Page size must be divisible by thread per block");
static_assert(GIVoxelPages::SegmentPerBlock != 0, "Segment should be bigger(or equal) than block");
static_assert(GIVoxelPages::SegmentSize < 2048, "Segment size should fit on SegmentPacked Structure, requires 10 bits at most");