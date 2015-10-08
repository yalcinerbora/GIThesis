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
	, vaoNormPosData(512)
	, vaoColorData(512)
{}

GICudaVoxelScene::~GICudaVoxelScene()
{
	if(vaoNormPosResource) CUDA_CHECK(cudaGraphicsUnregisterResource(vaoNormPosResource));
	if (vaoRenderResource) CUDA_CHECK(cudaGraphicsUnregisterResource(vaoRenderResource));
}

void GICudaVoxelScene::InitCuda()
{
	// Setting Device
	cudaSetDevice(0);

	// Cuda Check
	cudaDeviceProp props;
	CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

	// Info Print
	GI_LOG("Cuda Information...");
	GI_LOG("GPU Name\t\t: %s", props.name);
	GI_LOG("GPU Compute Capability\t: %d%d", props.major, props.minor);
	GI_LOG("GPU Shared Memory(SM)\t: %dKB", props.sharedMemPerMultiprocessor / 1024);
	GI_LOG("GPU Shared Memory(Block): %dKB", props.sharedMemPerBlock / 1024);
	GI_LOG("");

	// Minimum Required Compute Capability
	if(props.major < 3)
	{
		GI_LOG("#######################################################################");
		GI_LOG("UNSUPPORTED GPU, CUDA PORTION WILL NOT WORK. NEEDS ATLEAST SM_30 DEVICE");
		GI_LOG("#######################################################################");
		GI_LOG("");
	}

	// Shared Memory Prep
	// 16 Kb memory is enough for our needs most of the time
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// Voxel Transform Function needs 48kb memory
	CUDA_CHECK(cudaFuncSetCacheConfig(VoxelTransform, cudaFuncCachePreferShared));

}

void GICudaVoxelScene::LinkOGL(GLuint aabbBuffer,
							   GLuint transformBufferID,
							   GLuint infoBufferID,
							   GLuint voxelCacheNormPos,
							   GLuint voxelCacheIds,
							   GLuint voxelCacheRender,
							   uint32_t objCount,
							   uint32_t voxelCount)
{
	allocator.LinkOGLVoxelCache(aabbBuffer, transformBufferID, infoBufferID, 
								voxelCacheNormPos, voxelCacheIds, voxelCacheRender, 
								objCount, voxelCount);
}

void GICudaVoxelScene::AllocateWRTLinkedData(float coverageRatio)
{
	// Hint Device that we will use already linked resources
	allocator.ReserveForSegments(coverageRatio);
	vaoNormPosData.Resize(allocator.NumPages() * GI_PAGE_SIZE);
	vaoColorData.Resize(allocator.NumPages() * GI_PAGE_SIZE);

	// Cuda Register	
	if(vaoNormPosResource) CUDA_CHECK(cudaGraphicsUnregisterResource(vaoNormPosResource));
	if(vaoRenderResource) CUDA_CHECK(cudaGraphicsUnregisterResource(vaoRenderResource));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoNormPosResource, vaoNormPosData.getGLBuffer(), cudaGraphicsMapFlagsWriteDiscard));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&vaoRenderResource, vaoColorData.getGLBuffer(), cudaGraphicsMapFlagsWriteDiscard));
}

void GICudaVoxelScene::Reset()
{
	allocator.ResetSceneData();
}

void GICudaVoxelScene::VoxelUpdate(double& ioTiming,
								   double& updateTiming,
								   double& svoReconTiming,
								   const IEVector3& playerPos)
{
	// Pass if there is not any linked objects
	if(allocator.NumSegments() == 0) return;

	// Main Call Chain Called Every Frame
	// Manages Voxel Pages
	allocator.SetupDevicePointers();
	
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
			 allocator.GetTransformsDevice(i));
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
			 allocator.GetObjectAABBDevice(i),
			 allocator.GetTransformsDevice(i),
			 allocator.GetObjectInfoDevice(i),
			 
			 // Per Voxel Related
			 allocator.GetObjCacheIdsDevice(i),
			 allocator.NumVoxels(i),

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
			allocator.GetTransformsDevice(i));
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

	//-----------------------------------------------
	//DEBUG
	// ONLY WORKS IF THERE IS SINGLE SEGMENT IN THE SYSTEM
	// Call Logic Per Obj Segment
	unsigned int gridSize2 = (allocator.NumObjectSegments(0) + GI_THREAD_PER_BLOCK - 1) /
		GI_THREAD_PER_BLOCK;
	// KC DEBUG CHECK UNIQUE ALLOC
	DebugCheckUniqueAlloc<<<gridSize2, GI_THREAD_PER_BLOCK>>>(allocator.GetSegmentAllocLoc(0),
															  allocator.NumObjectSegments(0));
	CUDA_KERNEL_CHECK();
	// KC DEBUG CHECK UNIQUE SEGMENT ALLOC
	DebugCheckSegmentAlloc<<<gridSize2, GI_THREAD_PER_BLOCK>>>
		(*allocator.GetVoxelGridDevice(),
		allocator.GetSegmentAllocLoc(0),
		allocator.GetSegmentObjectID(0),
		allocator.NumObjectSegments(0),
		allocator.GetObjectAABBDevice(0),
		allocator.GetTransformsDevice(0));
	CUDA_KERNEL_CHECK();
	//DEBUG END
	//-----------------------------------------------

	timer.Stop();
	ioTiming = timer.ElapsedMilliS();

	// Now Call Update
	timer.Start();
	IEVector3 gridNewPos = allocator.GetNewVoxelPos(playerPos);

	gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;
	VoxelTransform<<<gridSize, GI_THREAD_PER_BLOCK>>>
	  (// Voxel Pages
	   allocator.GetVoxelPagesDevice(),
	   *allocator.GetVoxelGridDevice(),
	   float3{gridNewPos.getX(), gridNewPos.getY(), gridNewPos.getZ()},
	   
	   // Per Object Segment
	   allocator.GetSegmentAllocLoc2D(),				   

	   // Object Related
	   allocator.GetObjectAllocationIndexLookup2D(),
	   allocator.GetTransformsDevice(),
	   allocator.GetObjCacheNormPosDevice(),
	   allocator.GetObjRenderCacheDevice(),
	   allocator.GetObjectInfoDevice(),
	   allocator.GetObjectAABBDevice());
	CUDA_KERNEL_CHECK();

	allocator.SendNewVoxPosToDevice();
	
	timer.Stop();
	updateTiming = timer.ElapsedMilliS();

	// Then Call SVO Reconstruct
	timer.Start();
	timer.Stop();
	svoReconTiming = timer.ElapsedMilliS();

	// Done
	allocator.ClearDevicePointers();
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

	// KC VOXEL COUNT DETERMINE FROM VOXELS
	DetermineTotalVoxCount<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(*d_VoxCount,
		 // Page Related
		 allocator.GetVoxelPagesDevice(),
		 *allocator.GetVoxelGridDevice(),
		 allocator.NumPages());
	CUDA_KERNEL_CHECK();

	CUDA_CHECK(cudaMemcpy(&h_VoxCount, d_VoxCount, sizeof(int), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_VoxCount));
	return static_cast<uint32_t>(h_VoxCount);
}


VoxelDebugVAO GICudaVoxelScene::VoxelDataForRendering(CVoxelGrid& voxGridData, double& time, uint32_t voxCount)
{
	// Pass if there is not any linked objects
	if(allocator.NumSegments() > 0)
	{
		CudaTimer timer(0);
		timer.Start();

		// Map
		unsigned int* d_atomicCounter = nullptr;
		CVoxelNormPos* vBufferNormPosPtr = nullptr;
		uchar4* vBufferRenderPackedPtr = nullptr;
		size_t size = 0;

		allocator.SetupDevicePointers();

		unsigned int zero = 0;
		glBindBuffer(GL_COPY_WRITE_BUFFER, vaoNormPosData.getGLBuffer());
		glClearBufferData(GL_COPY_WRITE_BUFFER, GL_RG32UI, GL_RED, GL_UNSIGNED_INT, &zero);

		CUDA_CHECK(cudaGraphicsMapResources(1, &vaoNormPosResource));
		CUDA_CHECK(cudaGraphicsMapResources(1, &vaoRenderResource));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vBufferNormPosPtr), &size, vaoNormPosResource));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vBufferRenderPackedPtr), &size, vaoRenderResource));
	
		CUDA_CHECK(cudaMalloc(&d_atomicCounter, sizeof(unsigned int)));
		CUDA_CHECK(cudaMemset(d_atomicCounter, 0x00, sizeof(unsigned int)));

		// Copy
		// All Pages
		uint32_t gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
		VoxelCopyToVAO<<<gridSize, GI_THREAD_PER_BLOCK>>>
			(// Two ogl Buffers for rendering used voxels
			vBufferNormPosPtr,
			vBufferRenderPackedPtr,
			*d_atomicCounter,
			voxCount,

			// Per Obj Segment
			allocator.GetSegmentAllocLoc2D(),

			// Per obj
			allocator.GetObjectAllocationIndexLookup2D(),

			// Per vox
			allocator.GetObjRenderCacheDevice(),

			// Page
			allocator.GetVoxelPagesDevice(),
			allocator.NumPages(),
			*allocator.GetVoxelGridDevice());
		CUDA_KERNEL_CHECK();

		// Unmap
		cudaGraphicsUnmapResources(1, &vaoNormPosResource);
		cudaGraphicsUnmapResources(1, &vaoRenderResource);
		cudaFree(d_atomicCounter);
		allocator.ClearDevicePointers();

		timer.Stop();
		time = timer.ElapsedMilliS();

		voxGridData = allocator.GetVoxelGridHost();
	}
	return VoxelDebugVAO(vaoNormPosData, vaoColorData);
}