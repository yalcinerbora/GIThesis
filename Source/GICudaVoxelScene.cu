#include "GICudaVoxelScene.h"
#include "GIKernels.cuh"
#include "IEUtility/IEMath.h"
#include "CudaTimer.h"
#include "Macros.h"
#include "VoxelCopyToVAO.cuh"
#include "IEUtility/IEVector3.h"
#include "CDebug.cuh"
#include <cuda_gl_interop.h>

GICudaVoxelScene::GICudaVoxelScene(const IEVector3& intialCenterPos, float span, unsigned int dim)
	: allocator(CVoxelGrid 
				{
					{ 
						intialCenterPos.getX() - (dim * span * 0.5f),
						intialCenterPos.getY() - (dim * span * 0.5f),
						intialCenterPos.getZ() - (dim * span * 0.5f)
					},
					span, 
					{ dim, dim, dim }, 
					static_cast<unsigned int>(log2f(static_cast<float>(dim)))
				})
{}

GICudaVoxelScene::GICudaVoxelScene(GICudaVoxelScene&& other)
	: allocator(std::move(other.allocator))
{}

GICudaVoxelScene::~GICudaVoxelScene()
{}

void GICudaVoxelScene::LinkOGL(GLuint aabbBuffer,
							   GLuint transformBuffer,
							   GLuint transformIDBuffer,
							   GLuint infoBufferID,
							   GLuint voxelCacheNormPos,
							   GLuint voxelCacheIds,
							   GLuint voxelCacheRender,
							   uint32_t objCount,
							   uint32_t voxelCount)
{
	allocator.LinkOGLVoxelCache(aabbBuffer, transformBuffer, transformIDBuffer, infoBufferID,
								voxelCacheNormPos, voxelCacheIds, voxelCacheRender, 
								objCount, voxelCount);
}

void GICudaVoxelScene::AllocateWRTLinkedData(float coverageRatio)
{
	// Hint Device that we will use already linked resources
	allocator.ReserveForSegments(coverageRatio);
}

void GICudaVoxelScene::Reset()
{
	allocator.ResetSceneData();
}

void GICudaVoxelScene::VoxelUpdate(double& ioTiming,
								   double& updateTiming,
								   const IEVector3& playerPos,
								   float cascadeMultiplier)
{
	// Pass if there is not any linked objects
	if(allocator.NumSegments() == 0) return;

	// Manages Voxel Pages
	assert(allocator.IsGLMapped() == true);
	
	// Main Call Chain Called Every Frame
	CudaTimer timer(0);
	timer.Start();
	for(unsigned int i = 0; i < allocator.NumObjectBatches(); i++)
	{
		// Call Logic Per Obj Segment
		unsigned int gridSize = (allocator.NumObjectSegments(i) + GI_THREAD_PER_BLOCK_SMALL - 1) /
									GI_THREAD_PER_BLOCK_SMALL;

		// KC ALLOCATE
		VoxelObjectAlloc<<<gridSize, GI_THREAD_PER_BLOCK_SMALL>>>
			(// Voxel System
			 allocator.GetVoxelPagesDevice(),
			 allocator.NumPages(),
			 *allocator.GetVoxelGridDevice(),
			 
			 // Per Object Segment Related
			 allocator.GetSegmentAllocLoc(i),
			 allocator.GetSegmentObjectID(i),
			 allocator.NumObjectSegments(i),
			 
			 // Per Object Related
			 allocator.GetWriteSignals(i),
			 allocator.GetObjectAABBDevice(i),
			 allocator.GetTransformsDevice(i),
			 allocator.GetTransformIDDevice(i));
		CUDA_KERNEL_CHECK();

		// Call Logic Per Voxel
		gridSize = (allocator.NumVoxels(i) + GI_THREAD_PER_BLOCK - 1) /
					GI_THREAD_PER_BLOCK;
		
		// KC OBJECT VOXEL INCLUDE
		VoxelObjectInclude<<<gridSize, GI_THREAD_PER_BLOCK>>>
			(// Voxel System
			 allocator.GetVoxelPagesDevice(),
			 *allocator.GetVoxelGridDevice(),
			 
			 // Per Object Segment Related
			 allocator.GetSegmentAllocLoc(i),
			 allocator.NumObjectSegments(i),
			 
			 // Per Object Related
			 allocator.GetWriteSignals(i),
			 allocator.GetVoxelStrides(i),
			 allocator.GetObjectAllocationIndexLookup(i),
			 			 
			 // Per Voxel Related
			 allocator.GetObjCacheIdsDevice(i),
			 allocator.NumVoxels(i),
			 allocator.NumObjects(i),

			 // Batch(ObjectGroup in terms of OGL) Id
			 i);
		CUDA_KERNEL_CHECK();

		// Clear Write Signals
		CUDA_CHECK(cudaMemset(allocator.GetWriteSignals(i), 0, sizeof(char) * allocator.NumObjects(i)));
	}

	for(unsigned int i = 0; i < allocator.NumObjectBatches(); i++)
	{
		//timerSub.Start();

		// Call Logic Per Obj Segment
		unsigned int gridSize = (allocator.NumObjectSegments(i) + GI_THREAD_PER_BLOCK_SMALL - 1) /
			GI_THREAD_PER_BLOCK_SMALL;

		// KC DEALLOCATE
		VoxelObjectDealloc<<<gridSize, GI_THREAD_PER_BLOCK_SMALL>>>
			(// Voxel System
			allocator.GetVoxelPagesDevice(),
			*allocator.GetVoxelGridDevice(),

			// Per Object Segment Related
			allocator.GetSegmentAllocLoc(i),
			allocator.GetSegmentObjectID(i),
			allocator.NumObjectSegments(i),

			// Per Object Related
			allocator.GetWriteSignals(i),
			allocator.GetObjectAABBDevice(i),
			allocator.GetTransformsDevice(i),
			allocator.GetTransformIDDevice(i));
		CUDA_KERNEL_CHECK();
	}

	// Call Logic Per Voxel in Page
	unsigned int gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
							GI_THREAD_PER_BLOCK;

	// KC CLEAR MARKED
	VoxelClearMarked<<<gridSize, GI_THREAD_PER_BLOCK>>>(allocator.GetVoxelPagesDevice());
	CUDA_KERNEL_CHECK();

	// Call Logic Per Segment in Page
	gridSize = (allocator.NumPages() * GI_SEGMENT_PER_PAGE + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;

	// KC CLEAR SIGNAL
	VoxelClearSignal<<<gridSize, GI_THREAD_PER_BLOCK>>>(allocator.GetVoxelPagesDevice(),
														allocator.NumPages());
	CUDA_KERNEL_CHECK();

	////-----------------------------------------------
	////DEBUG
	//// ONLY WORKS IF THERE IS SINGLE SEGMENT IN THE SYSTEM
	//// Call Logic Per Obj Segment
	//unsigned int gridSize2 = (allocator.NumObjectSegments(0) + GI_THREAD_PER_BLOCK - 1) /
	//	GI_THREAD_PER_BLOCK;
	//// KC DEBUG CHECK UNIQUE ALLOC
	//DebugCheckUniqueAlloc<<<gridSize2, GI_THREAD_PER_BLOCK>>>(allocator.GetSegmentAllocLoc(0),
	//														  allocator.NumObjectSegments(0));
	//CUDA_KERNEL_CHECK();
	//// KC DEBUG CHECK UNIQUE SEGMENT ALLOC
	//DebugCheckSegmentAlloc<<<gridSize2, GI_THREAD_PER_BLOCK>>>
	//	(*allocator.GetVoxelGridDevice(),
	//	allocator.GetSegmentAllocLoc(0),
	//	allocator.GetSegmentObjectID(0),
	//	allocator.NumObjectSegments(0),
	//	allocator.GetObjectAABBDevice(0),
	//	allocator.GetTransformsDevice(0));
	//CUDA_KERNEL_CHECK();
	////DEBUG END
	////-----------------------------------------------

	timer.Stop();
	ioTiming = timer.ElapsedMilliS();

	// Now Call Update
	timer.Start();
	IEVector3 gridNewPos = allocator.GetNewVoxelPos(playerPos, cascadeMultiplier);

	gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;
	VoxelTransform<<<gridSize, GI_THREAD_PER_BLOCK>>>
	  (// Voxel Pages
	   allocator.GetVoxelPagesDevice(),
	   *allocator.GetVoxelGridDevice(),
	   float3{gridNewPos.getX(), gridNewPos.getY(), gridNewPos.getZ()},

	   // Object Related
	   allocator.GetTransformsDevice(),
	   allocator.GetTransformIDDevice(),
	   allocator.GetObjCacheNormPosDevice(),
	   allocator.GetObjRenderCacheDevice(),
	   allocator.GetObjectInfoDevice(),
	   allocator.GetObjectAABBDevice());
	CUDA_KERNEL_CHECK();	

	allocator.SendNewVoxPosToDevice();
	
	timer.Stop();
	updateTiming = timer.ElapsedMilliS();
}

uint64_t GICudaVoxelScene::AllocatorMemoryUsage() const
{
	return allocator.SystemTotalMemoryUsage();
}

uint32_t GICudaVoxelScene::VoxelCountInPage()
{
	// Pass if there is not any linked objects
	if(allocator.NumSegments() == 0) return 0;

	int h_VoxCount;
	int* d_VoxCount = nullptr;

	CUDA_CHECK(cudaMalloc(&d_VoxCount, sizeof(int)));
	CUDA_CHECK(cudaMemset(d_VoxCount, 0, sizeof(int)));

	uint32_t gridSize = ((allocator.NumPages() * GI_PAGE_SIZE) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	VoxCountPage<<<gridSize, GI_THREAD_PER_BLOCK>>>
	(
		*d_VoxCount,
		 // Page Related
		 allocator.GetVoxelPagesDevice(),
		 *allocator.GetVoxelGridDevice(),
		 allocator.NumPages()
	);
	CUDA_KERNEL_CHECK();

	CUDA_CHECK(cudaMemcpy(&h_VoxCount, d_VoxCount, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_VoxCount));
	return static_cast<uint32_t>(h_VoxCount);
}


double GICudaVoxelScene::VoxDataToGL(CVoxelNormPos* dVAONormPosData,
									 uchar4* dVAOColorData,

									 CVoxelGrid& voxGridData,
									 uint32_t& voxCount,
									 uint32_t maxVoxelCount)
{
	// Pass if there is not any linked objects
	if(allocator.NumSegments() > 0)
	{
		CudaTimer timer(0);
		timer.Start();
		
		unsigned int* d_atomicCounter = nullptr;

		allocator.SetupDevicePointers();		
		CUDA_CHECK(cudaMalloc(&d_atomicCounter, sizeof(unsigned int)));
		CUDA_CHECK(cudaMemset(d_atomicCounter, 0x00, sizeof(unsigned int)));

		// Copy
		// All Pages
		uint32_t gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
		VoxCpyPage<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(	// Two ogl Buffers for rendering used voxels
			dVAONormPosData,
			dVAOColorData,
			*d_atomicCounter,
			maxVoxelCount,

			// Per Obj Segment
			allocator.GetSegmentAllocLoc2D(),

			// Per obj
			allocator.GetObjectAllocationIndexLookup2D(),

			// Per vox
			allocator.GetObjRenderCacheDevice(),

			// Page
			allocator.GetVoxelPagesDevice(),
			allocator.NumPages(),
			*allocator.GetVoxelGridDevice()
		);
		CUDA_KERNEL_CHECK();

		// Copy Actual Vox Count
		CUDA_CHECK(cudaMemcpy(&voxCount, d_atomicCounter,
					sizeof(unsigned int), cudaMemcpyDeviceToHost));

		// Unmap
		cudaFree(d_atomicCounter);
		allocator.ClearDevicePointers();
		voxGridData = allocator.GetVoxelGridHost();

		timer.Stop();
		return timer.ElapsedMilliS();
	}
	return 0.0;
}

GICudaAllocator* GICudaVoxelScene::Allocator()
{
	return &allocator;
}

void GICudaVoxelScene::MapGLPointers()
{
	allocator.SetupDevicePointers();
}

void GICudaVoxelScene::UnmapGLPointers()
{
	allocator.ClearDevicePointers();
}