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
		// Class that handles debug rendering of the page
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
													 VoxelRenderType renderType,
													 const Camera& camera,
													 const GIVoxelCache& cache,
													 const GIVoxelPages& pages,
													 bool useCache);
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

		// OGL Buffer Resources (Model, Transform Index, AABB)
		std::vector<cudaGraphicsResource_t>		batchOGLResources;
		
		uint16_t								PackSegmentInfo(const uint8_t cascadeId,
																const CObjectType type,
																const CSegmentOccupation occupation,
																const bool firstOccurance);
		void									GenerateGPUData(const GIVoxelCache& cache);
		void									AllocatePages(size_t voxelCapacity);

		// Update Functions; should be called in this order		
		void									UpdateGridPositions(const IEVector3& cameraPos);
		void									MapOGLResources();
		double									VoxelIO(bool doTiming);
		double									Transform(const GIVoxelCache& cache,
														  bool doTiming);
		void									UnmapOGLResources();

	protected:
		// Batch
		const std::vector<MeshBatchI*>*			batches;
		const OctreeParameters*					svoParams;
		uint32_t								segmentAmount;
		IEVector3								outermostGridPosition;

		//Page System (Theoretically Dynamic Data)
		std::vector<MultiPage>					hPages;
		CudaVector<CVoxelPage>					dPages;

		// Debug Rednering Related
		PageRenderer							pageRenderer;

		void									GenerateGridPositions(std::vector<IEVector3>& gridPositions,
																	  const IEVector3& cameraPos);

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

		virtual void							Update(double& ioTime,
													   double& transTime,
													   const GIVoxelCache& caches,
													   const IEVector3& camPos,
													   bool doTiming);
		
		uint64_t								MemoryUsage() const;
		uint32_t								PageCount() const;

		// Debug File
		void									DumpPageSegments(const char*, size_t offset = 0, size_t pageCount = 0) const;
		void									DumpPageEmptyPositions(const char*, size_t offset = 0, size_t pageCount = 0) const;
		void									DumpSegmentAllocation(const char*, size_t offset = 0, size_t segmentCount =  0) const;
		void									DumpSegmentInfo(const char*, size_t offset = 0, size_t segmentCount = 0) const;

		// Debug Draw
		void									AllocateDraw();
		virtual double							Draw(bool doTiming, 
													 uint32_t cascade,
													 VoxelRenderType renderType,
													 const Camera& camera,
													 const GIVoxelCache& cache);
		void									DeallocateDraw();

		const CVoxelPageConst*					getVoxelPagesDevice() const;
		const CVoxelGrid*						getVoxelGridsDevice() const;
		const IEVector3&						getOutermostGridPosition() const;
		size_t									VoxelCountInCirculation(const GIVoxelCache& cache) const;
};

static_assert(GIVoxelPages::PageSize % CudaInit::TBP == 0, "Page size must be divisible by thread per block");
static_assert(GIVoxelPages::SegmentPerBlock != 0, "Segment should be bigger(or equal) than a block");
static_assert(GIVoxelPages::SegmentSize <= 1024, "Segment size should fit on SegmentPacked Structure, requires 10 bits at most");

class GIVoxelPagesFrame : public GIVoxelPages
{
	private:
		class FastVoxelizer
		{
			private:
				// Data Buffer that is used for fast voxelization
				StructuredBuffer<uint8_t>	oglData;
				GLuint						lockTexture;
				cudaGraphicsResource_t		denseResource;				
				// Offsets
				size_t						incrementOffset;
				size_t						denseOffset;

				const OctreeParameters*		octreeParams;

				// Shaders
				Shader						vertVoxelizeFast;
				Shader						vertVoxelizeFastSkeletal;
				Shader						geomVoxelize;
				Shader						fragVoxelizeFast;
				
				double						Voxelize(const std::vector<MeshBatchI*>& batches,
													 const IEVector3& gridCorner, float span,
													 bool doTiming);
				double						Filter(uint32_t& voxelCount,
												   uint32_t segmentOffset,
												   CudaVector<CVoxelPage>& dVoxelPages,
												   uint32_t cascadeId,
												   bool doTiming);

			protected:
			public:
				// Constructors & Destructor
											FastVoxelizer();
											FastVoxelizer(const OctreeParameters*);
											FastVoxelizer(const FastVoxelizer&) = delete;
											FastVoxelizer(FastVoxelizer&&);
				FastVoxelizer&				operator=(const FastVoxelizer&) = delete;
				FastVoxelizer&				operator=(FastVoxelizer&&);
											~FastVoxelizer();

				// Map
				double						FastVoxelize(uint32_t& usedSegmentCount,
														 CudaVector<CVoxelPage>& dVoxelPages,
														 const std::vector<MeshBatchI*>& batches,
														 const std::vector<IEVector3>& gridPositions,								 
														 bool doTiming);
		};

	private:
		// Comparison between hardware rasterizer based voxelization
		FastVoxelizer						fastVoxelizer;
		uint32_t							usedSegmentCount;
		
	protected:
	public:
											// Constrcutors & Destructor
											GIVoxelPagesFrame() = default;
											GIVoxelPagesFrame(const GIVoxelCache& cache,
															  const std::vector<MeshBatchI*>* batches,
															  const OctreeParameters& octreeParams);
											GIVoxelPagesFrame(const GIVoxelPagesFrame&) = delete;
											GIVoxelPagesFrame(GIVoxelPagesFrame&&) = default;
		GIVoxelPagesFrame&					operator=(const GIVoxelPagesFrame&) = delete;
		GIVoxelPagesFrame&					operator=(GIVoxelPagesFrame&&) = default;
											~GIVoxelPagesFrame() = default;


		void								ClearPages();
		virtual void						Update(double& ioTime,
												   double& transTime,
												   const GIVoxelCache& caches,
												   const IEVector3& camPos,
												   bool doTiming) override;
		virtual double						Draw(bool doTiming,
												 uint32_t cascadeCount,
												 VoxelRenderType renderType,
												 const Camera& camera,
												 const GIVoxelCache& cache) override;
};