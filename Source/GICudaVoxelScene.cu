#include "GICudaVoxelScene.h"
#include "GIKernels.cuh"
#include "IEUtility/IEMath.h"
#include "CudaTimer.h"
#include "Macros.h"
#include "VoxelCopyToVAO.cuh"
#include "IEUtility/IEVector3.h"

GICudaVoxelScene::GICudaVoxelScene(const CVoxelGrid& gridSetup)
	: allocator(gridSetup)
	, vaoData(512)
	, vaoRenderData(512)
	, voxelVAO(vaoData, vaoRenderData)
{}

GICudaVoxelScene::~GICudaVoxelScene()
{}

void GICudaVoxelScene::LinkOGL(GLuint aabbBuffer,
							   GLuint transformBufferID,
							   GLuint relativeTransformBufferID,
							   GLuint infoBufferID,
							   GLuint voxelCache,
							   GLuint voxelCacheRender,
							   uint32_t objCount,
							   uint32_t voxelCount)
{
	allocator.LinkOGLVoxelCache(aabbBuffer, transformBufferID, relativeTransformBufferID,
								infoBufferID, voxelCache, voxelCacheRender, objCount, voxelCount);
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
	vaoRenderData.Resize(voxCount);
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

		// KC OBJECT VOXEL EXLCUDE
		VoxelObjectExclude<<<gridSize, GI_THREAD_PER_BLOCK>>>	
			(// Voxel Pages
			 allocator.GetVoxelPagesDevice(),
			 allocator.NumPages(),
			 *allocator.GetVoxelGridDevice(),

			 // Per Object Segment
			 allocator.GetSegmentAllocLoc(i),
			 allocator.GetSegmentObjectID(i),
			 allocator.NumObjectSegments(i),

			 // Per Object
			 allocator.GetObjectAABBDevice(i),
			 allocator.GetTransformsDevice(i));
		
		// Call Logic Per Object Segment
		gridSize = (allocator.NumObjectSegments(i) + GI_THREAD_PER_BLOCK - 1) /
			GI_THREAD_PER_BLOCK;

		// KC ALLOCATE
		VoxelObjectAllocate<<<gridSize, GI_THREAD_PER_BLOCK>>>
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
			 allocator.NumPages(),
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
	}
	timer.Stop();
	ioTiming = timer.ElapsedMilliS();
	
	// Now Call Update
	timer.Start();
	IEVector3 gridNewPos = allocator.GetNewVoxelPos(playerPos);

	timer.Stop();
	updateTiming = timer.ElapsedMilliS();

	// Then Call SVO Reconstruct
	timer.Start();
	timer.Stop();
	svoReconTiming = timer.ElapsedMilliS();


	// Done
	cudaDeviceSynchronize();
	allocator.ClearDevicePointers();
}

uint32_t GICudaVoxelScene::VoxelCountInPage()
{
	int h_VoxCount;
	int* d_VoxCount = nullptr;

	cudaMalloc(&d_VoxCount, sizeof(int));
	cudaMemset(d_VoxCount, 0, sizeof(int));

	for(unsigned int i = 0; i < allocator.NumObjectBatches(); i++)
	{
		uint32_t gridSize = (allocator.NumObjects(i) + GI_THREAD_PER_BLOCK - 1) / GI_THREAD_PER_BLOCK;

		// KC VOXEL COUNT DETERMINE
		DetermineTotalVoxCount<<<gridSize, GI_THREAD_PER_BLOCK>>>
			(*d_VoxCount,
			 *allocator.GetVoxelGridDevice(),
			 
			 // Per Obj Segment
			 allocator.GetSegmentAllocLoc(i),
			 
			 // Per obj
			 allocator.GetObjectAllocationIndexLookup(i),
			 allocator.GetObjectInfoDevice(i),
			 allocator.GetTransformsDevice(i),
			 allocator.NumObjects(i),
			 allocator.NumObjectSegments(i));
	}
	cudaMemcpy(&h_VoxCount, d_VoxCount, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_VoxCount);
	return static_cast<uint32_t>(h_VoxCount);
}


VoxelDebugVAO& GICudaVoxelScene::VoxelDataForRendering(uint32_t voxCount)
{
	

	//// KC OBJECT VOXEL INCLUDE
	//VoxelObjectInclude << <gridSize, GI_THREAD_PER_BLOCK >> >
	//	(// Voxel System
	//	allocator.GetVoxelPagesDevice(),
	//	allocator.NumPages(),
	//	*allocator.GetVoxelGridDevice(),

	//	// Per Object Segment Related
	//	allocator.GetSegmentAllocLoc(i),
	//	allocator.GetSegmentObjectID(i),
	//	allocator.NumObjectSegments(i),

	//	// Per Object Related
	//	allocator.GetWriteSignals(i),
	//	allocator.GetVoxelStrides(i),
	//	allocator.GetObjectAllocationIndexLookup(i),
	//	allocator.GetObjectAABBDevice(i),
	//	allocator.GetTransformsDevice(i),
	//	allocator.GetObjectInfoDevice(i),
	//	allocator.NumObjects(i),

	//	// Per Voxel Related
	//	allocator.GetObjCacheDevice(i),
	//	allocator.NumVoxels(i),

	//	// Batch(ObjectGroup in terms of OGL) Id
	//	i);

	
	// Determine the size of vao buffer
	//voxCount = 0;
		// KC
	// Resize structured buffers
	// Copy to GL Buffer
		// KC


	
	return voxelVAO;
}