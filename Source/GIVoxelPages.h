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
class GIVoxelCache;

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

		uint32_t								segmentSize;

		// Static GPU Data
		CudaVector<uint8_t>						gpuData;
		// All these pointers are offseted on the gpuData
		// Grid Related
		CVoxelGrid*								dVoxelGrids;
		float3*									dNewGridPositions;
		// OGL Pointer Data
		BatchOGLData*							dBatchOGLData;
		// Helper Data
		CSegmentInfo*							dSegmentInfo;
		ushort2*								dSegmentAllocInfo;

		//Page System (Theoretically Dynamic Data)
		std::vector<MultiPage>					hPages;
		CudaVector<CVoxelPage>					dPages;
		
		// OGL Related
		std::vector<cudaGraphicsResource_t>		batchOGLResources;
		

		void									AllocatePages(size_t voxelCapacity);
		void									UpdateGridPositions(const IEVector3& cameraPos);
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
		double									Transform(const GIVoxelCache& cache,
														  const IEVector3 cameraPos);
	
		// Debug Draw
		void									AllocateDraw();
		void									Draw(size_t cascadeCount);
		void									DeallocateDraw();

		const CVoxelPageConst*					getVoxelPages() const;
		const CVoxelPage*						getVoxelPages();
};

static_assert(GIVoxelPages::PageSize % CudaInit::TBP == 0, "Page size must be divisible by thread per block");
static_assert(GIVoxelPages::SegmentPerBlock != 0, "Segment should be bigger(or equal) than block");
static_assert(GIVoxelPages::SegmentSize < 2048, "Segment size should fit on SegmentPacked Structure, requires 10 bits at most");