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
	, vaoData(512)
	, vaoColorData(512)
{

	// Shared Memory Prep
	// 16 Kb memory is enough for our needs most of the time
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Voxel Transform Function needs 48kb memory
	cudaFuncSetCacheConfig(VoxelTransform, cudaFuncCachePreferShared);

}

GICudaVoxelScene::~GICudaVoxelScene()
{
	cudaGraphicsUnregisterResource(vaoResource);
	cudaGraphicsUnregisterResource(vaoRenderResource);
}

void GICudaVoxelScene::LinkOGL(GLuint aabbBuffer,
							   GLuint transformBufferID,
							   GLuint infoBufferID,
							   GLuint voxelCache,
							   GLuint voxelCacheRender,
							   uint32_t objCount,
							   uint32_t voxelCount)
{
	allocator.LinkOGLVoxelCache(aabbBuffer, transformBufferID, infoBufferID, 
								voxelCache, voxelCacheRender, objCount, voxelCount);
}

void GICudaVoxelScene::LinkSceneTextures(GLuint shadowMapArray)
{
	allocator.LinkSceneShadowMapArray(shadowMapArray);
}

void GICudaVoxelScene::UnLinkDeferredRendererBuffers()
{
	allocator.UnLinkGBuffers();
}

void GICudaVoxelScene::LinkDeferredRendererBuffers(GLuint depthBuffer,
												   GLuint normalGBuff,
												   GLuint lightIntensityTex)
{
	allocator.LinkSceneGBuffers(depthBuffer, normalGBuff, lightIntensityTex);
}

void GICudaVoxelScene::AllocateInitialPages(uint32_t approxVoxCount)
{
	// Hint Device that we will use already linked resources
	uint32_t voxCount = IEMath::UpperPowTwo(static_cast<unsigned int>(approxVoxCount));
	uint32_t pageCount = (voxCount + (GI_PAGE_SIZE - 1)) / GI_PAGE_SIZE;
	allocator.Reserve(pageCount);
	vaoData.Resize(voxCount);
	vaoColorData.Resize(voxCount);

	// Cuda Register	
	cudaGraphicsGLRegisterBuffer(&vaoResource, vaoData.getGLBuffer(), cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsGLRegisterBuffer(&vaoRenderResource, vaoColorData.getGLBuffer(), cudaGraphicsMapFlagsWriteDiscard);
}

void GICudaVoxelScene::Reset()
{
	allocator.ResetSceneData();
}

void GICudaVoxelScene::Voxelize(double& ioTiming,
								double& updateTiming,
								double& svoReconTiming,
								const IEVector3& playerPos)
{
	CudaTimer timer(0);
	timer.Start();

	// Main Call Chain Called Every Frame
	// Manages Voxel Pages
	allocator.SetupDevicePointers();

	for(unsigned int i = 0; i < allocator.NumObjectBatches(); i++)
	{
		// Call Logic Per Obj Segment
		unsigned int gridSize = (allocator.NumObjectSegments(i) + GI_THREAD_PER_BLOCK - 1) /
			GI_THREAD_PER_BLOCK;

		// KC ALLOCATE
		VoxelObjectAlloc<<<gridSize, GI_THREAD_PER_BLOCK>>>
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
			 allocator.GetObjCacheDevice(i),
			 allocator.NumVoxels(i),

			 // Batch(ObjectGroup in terms of OGL) Id
			 i);

		// Clear Write Signals
		cudaMemset(allocator.GetWriteSignals(i), 0, sizeof(char) * allocator.NumObjects(i));
	}

	for(unsigned int i = 0; i < allocator.NumObjectBatches(); i++)
	{
		// Call Logic Per Obj Segment
		unsigned int gridSize = (allocator.NumObjectSegments(i) + GI_THREAD_PER_BLOCK - 1) /
			GI_THREAD_PER_BLOCK;

		// KC DEALLOCATE
		VoxelObjectDealloc<<<gridSize, GI_THREAD_PER_BLOCK>>>
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
	}

	// Call Logic Per Voxel in Page
	unsigned int gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
							GI_THREAD_PER_BLOCK;

	// KC CLEAR MARKED
	VoxelClearMarked<<<gridSize, GI_THREAD_PER_BLOCK>>>(allocator.GetVoxelPagesDevice());

	// Call Logic Per Segment in Page
	gridSize = (allocator.NumPages() * GI_SEGMENT_PER_PAGE + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;

	// KC CLEAR SIGNAL
	VoxelClearSignal<<<gridSize, GI_THREAD_PER_BLOCK>>>(allocator.GetVoxelPagesDevice());


	////DEBUG
	//// ONLY WORKS IF THERE IS SINGLE SEGMENT IN THE SYSTEM
	//// Call Logic Per Obj Segment
	//unsigned int gridSize2 = (allocator.NumObjectSegments(0) + GI_THREAD_PER_BLOCK - 1) /
	//	GI_THREAD_PER_BLOCK;
	//// KC DEBUG CHECK UNIQUE ALLOC
	//DebugCheckUniqueAlloc<<<gridSize2, GI_THREAD_PER_BLOCK>>>(allocator.GetSegmentAllocLoc(0),
	//														  allocator.NumObjectSegments(0));
	//// KC DEBUG CHECK UNIQUE SEGMENT ALLOC
	//DebugCheckSegmentAlloc<<<gridSize2, GI_THREAD_PER_BLOCK>>>
	//	(*allocator.GetVoxelGridDevice(),
	//	allocator.GetSegmentAllocLoc(0),
	//	allocator.GetSegmentObjectID(0),
	//	allocator.NumObjectSegments(0),
	//	allocator.GetObjectAABBDevice(0),
	//	allocator.GetTransformsDevice(0));
	////DEBUG END






	timer.Stop();
	ioTiming = timer.ElapsedMilliS();

	// Now Call Update
	timer.Start();
	IEVector3 gridNewPos = allocator.GetNewVoxelPos(playerPos);

	gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) /
				GI_THREAD_PER_BLOCK;
	VoxelTransform <<<gridSize, GI_THREAD_PER_BLOCK>>>
	  (// Voxel Pages
	   allocator.GetVoxelPagesDevice(),
	   *allocator.GetVoxelGridDevice(),
	   float3{gridNewPos.getX(), gridNewPos.getY(), gridNewPos.getZ()},
	   
	   // Per Object Segment
	   allocator.GetSegmentAllocLoc2D(),				   

	   // Object Related
	   allocator.GetObjectAllocationIndexLookup2D(),
	   allocator.GetTransformsDevice(),
	   allocator.GetObjRenderCacheDevice(),
	   allocator.GetObjCacheDevice(),
	   allocator.GetObjectInfoDevice(),
	   allocator.GetObjectAABBDevice());

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

uint32_t GICudaVoxelScene::VoxelCountInPage()
{
	int h_VoxCount;
	int* d_VoxCount = nullptr;

	cudaMalloc(&d_VoxCount, sizeof(int));
	cudaMemset(d_VoxCount, 0, sizeof(int));

	uint32_t gridSize = ((allocator.NumPages() * GI_PAGE_SIZE) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;

	// KC VOXEL COUNT DETERMINE FROM VOXELS
	DetermineTotalVoxCount<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(*d_VoxCount,
		 // Page Related
		 allocator.GetVoxelPagesDevice(),
		 *allocator.GetVoxelGridDevice(),
		 allocator.NumPages());

	cudaMemcpy(&h_VoxCount, d_VoxCount, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_VoxCount);
	return static_cast<uint32_t>(h_VoxCount);
}


VoxelDebugVAO GICudaVoxelScene::VoxelDataForRendering(CVoxelGrid& voxGridData, double& time, uint32_t voxCount)
{
	CudaTimer timer(0);
	timer.Start();

	// Map
	unsigned int* d_atomicCounter = nullptr;
	CVoxelPacked* vBufferPackedPtr = nullptr;
	uchar4* vBufferRenderPackedPtr = nullptr;
	size_t size = 0;

	allocator.SetupDevicePointers();

	unsigned int zero = 0;
	glBindBuffer(GL_COPY_WRITE_BUFFER, vaoData.getGLBuffer());
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_RG32UI, GL_RED, GL_UNSIGNED_INT, &zero);

	cudaGraphicsMapResources(1, &vaoResource);
	cudaGraphicsMapResources(1, &vaoRenderResource);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vBufferPackedPtr), &size, vaoResource);
	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vBufferRenderPackedPtr), &size, vaoRenderResource);
	
	cudaMalloc(&d_atomicCounter, sizeof(unsigned int));
	cudaMemset(d_atomicCounter, 0x00, sizeof(unsigned int));

	// Copy
	// All Pages
	uint32_t gridSize = (allocator.NumPages() * GI_PAGE_SIZE + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;
	VoxelCopyToVAO<<<gridSize, GI_THREAD_PER_BLOCK>>>
		(// Two ogl Buffers for rendering used voxels
		vBufferPackedPtr,
		vBufferRenderPackedPtr,
		*d_atomicCounter,

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
		
	voxGridData = allocator.GetVoxelGridHost();

	// Unmap
	cudaGraphicsUnmapResources(1, &vaoResource);
	cudaGraphicsUnmapResources(1, &vaoRenderResource);
	cudaFree(d_atomicCounter);
	allocator.ClearDevicePointers();

	timer.Stop();
	time = timer.ElapsedMilliS();

	return VoxelDebugVAO(vaoData, vaoColorData);
}