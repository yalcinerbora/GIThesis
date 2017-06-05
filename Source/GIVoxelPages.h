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
#include "VoxelVAO.h"
#include "Shader.h"

typedef uint2 CVoxelIds;

class SceneI;
class GIVoxelCache;

struct CModelTransform;
class OctreeParameters;

class GIVoxelPages
{
	private:
		class PageRenderer
		{
			private:
				Shader							vRenderWorldVoxel;
				Shader							fRenderWorldVoxel;

				// Buffer and its resource
				cudaGraphicsResource_t			debugBufferResource;		
				StructuredBuffer<uint8_t>		debugDrawBuffer;
				uint8_t*						debugBufferCUDA;

				// VAO
				VoxelVAO						debugDrawVao;
		
				// Offset
				size_t							drawParameterOffset;
				size_t							atomicIndexOffset;
				size_t							gridInfoOffset;
				size_t							voxelPositionOffset;
				size_t							voxelRenderOffset;

			protected:
			public:
				// Constructors & Destructor
												PageRenderer();
												PageRenderer(const GIVoxelPages&);
												PageRenderer(const PageRenderer&) = delete;
												PageRenderer(PageRenderer&&);
				PageRenderer&					operator=(const PageRenderer&) = delete;
				PageRenderer&					operator=(PageRenderer&&);
												~PageRenderer();

				double							Draw(bool doTiming,
													 uint32_t cascade,
													 VoxelRender renderType,
													 const Camera& camera,
													 const GIVoxelCache& cache,
													 const GIVoxelPages& pages);
				bool							Allocated() const;
		};

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
		const OctreeParameters*					svoParams;
		uint32_t								segmentAmount;

		// Static GPU Data
		CudaVector<uint8_t>						gpuData;
		// All these pointers are offseted on the gpuData
		// Grid Related
		CVoxelGrid*								dVoxelGrids;
		// OGL Pointer Data
		BatchOGLData*							dBatchOGLData;
		// Helper Data
		CSegmentInfo*							dSegmentInfo;
		ushort2*								dSegmentAllocInfo;

		//Page System (Theoretically Dynamic Data)
		std::vector<MultiPage>					hPages;
		CudaVector<CVoxelPage>					dPages;

		// OGL Buffer Resources (Model, Transform Index, AABB)
		std::vector<cudaGraphicsResource_t>		batchOGLResources;

		// Debug Rednering Related
		PageRenderer							pageRenderer;
		

		uint16_t								PackSegmentInfo(const uint8_t cascadeId,
																const CObjectType type,
																const CSegmentOccupation occupation);
		void									GenerateGPUData(const GIVoxelCache& cache);
		void									AllocatePages(size_t voxelCapacity);

	protected:
	public:
		// Constrcutors & Destructor
												GIVoxelPages();
												GIVoxelPages(const GIVoxelCache& cache,
															 const std::vector<MeshBatchI*>* batches,
															 const OctreeParameters& octreeParams);
												GIVoxelPages(const GIVoxelPages&) = delete;
												GIVoxelPages(GIVoxelPages&&);
		GIVoxelPages&							operator=(const GIVoxelPages&) = delete;
		GIVoxelPages&							operator=(GIVoxelPages&&);
												~GIVoxelPages();

		// Update Functions; should be called in this order		
		void									UpdateGridPositions(const IEVector3& cameraPos);
		void									MapOGLResources();
		double									VoxelIO(bool doTiming);
		double									Transform(const GIVoxelCache& cache,
														  bool doTiming);
		void									UnmapOGLResources();

		uint64_t								MemoryUsage() const;
	
		// Debug File
		void									DumpPageSegments(const char*, size_t offset = 0, size_t pageCount = 0) const;
		void									DumpPageEmptyPositions(const char*, size_t offset = 0, size_t pageCount = 0) const;
		void									DumpSegmentAllocation(const char*, size_t offset = 0, size_t segmentCount =  0) const;
		void									DumpSegmentInfo(const char*, size_t offset = 0, size_t segmentCount = 0) const;

		// Debug Draw
		void									AllocateDraw();
		double									Draw(bool doTiming, 
													 uint32_t cascade,
													 VoxelRender renderType,
													 const Camera& camera,
													 const GIVoxelCache& cache);
		void									DeallocateDraw();

		const CVoxelPageConst*					getVoxelPages() const;
		const CVoxelPage*						getVoxelPages();
};

static_assert(GIVoxelPages::PageSize % CudaInit::TBP == 0, "Page size must be divisible by thread per block");
static_assert(GIVoxelPages::SegmentPerBlock != 0, "Segment should be bigger(or equal) than block");
static_assert(GIVoxelPages::SegmentSize < 2048, "Segment size should fit on SegmentPacked Structure, requires 10 bits at most");