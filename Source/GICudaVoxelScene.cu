#include "GICudaVoxelScene.h"
#include "GIKernels.cuh"
#include "IEUtility/IEMath.h"

GICudaVoxelScene::GICudaVoxelScene()
	: dVoxGrid(nullptr)
	, allocator(CVoxelGrid { { 0.0f, 0.0f, 0.0f }, 1.0f, {512, 512 ,512},  9})
{
	cudaMalloc(reinterpret_cast<void**>(dVoxGrid), sizeof(CVoxelGrid));
}

GICudaVoxelScene::~GICudaVoxelScene()
{
	cudaFree(dVoxGrid);
}

void GICudaVoxelScene::LinkOGL(GLuint aabbBuffer,
							   GLuint transformBufferID,
							   GLuint relativeTransformBufferID,
							   GLuint infoBufferID,
							   GLuint voxelCache,
							   GLuint voxelCacheRender,
							   size_t objCount)
{
	allocator.LinkOGLVoxelCache(aabbBuffer, transformBufferID, relativeTransformBufferID,
								infoBufferID, voxelCache, voxelCacheRender, objCount);
}

void GICudaVoxelScene::LinkSceneTextures(const std::vector<GLuint>& shadowMaps)
{
	allocator.LinkSceneShadowMapArray(shadowMaps);
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
}

void GICudaVoxelScene::Reset()
{
	allocator.ResetSceneData();
}

void GICudaVoxelScene::Voxelize(const IEVector3& playerPos)
{
	// Main Call Chain Called Every Frame
	// Manages Voxel Pages

	allocator.SetupDevicePointers();


	//VoxelObjectInclude(// Voxel System
	//				   allocator.GetVoxelPagesDevice(),
	//				   allocator.NumPages(),
	//				   allocator.
	//				   
	//				   
	//				   const CVoxelGrid& gGridInfo,

	//				   // Per Object Segment Related
	//				   ushort2* gObjectAllocLocations,
	//				   unsigned int* gSegmentObjectId,
	//				   uint32_t totalSegments,

	//				   // Per Object Related
	//				   char* gWriteToPages,
	//				   const unsigned int* gObjectVoxStrides,
	//				   const unsigned int* gObjectAllocIndexLookup,
	//				   const CObjectAABB* gObjectAABB,
	//				   const CObjectTransform* gObjTransforms,
	//				   const CObjectVoxelInfo* gObjInfo,
	//				   uint32_t objectCount,

	//				   // Per Voxel Related
	//				   const CVoxelPacked* gObjectVoxelCache,
	//				   uint32_t voxCount,

	//				   // Batch(ObjectGroup in terms of OGL) Id
	//				   uint32_t batchId);


	

	allocator.ClearDevicePointers();
}

GLuint GICudaVoxelScene::VoxelDataForRendering()
{
	return 0;
}