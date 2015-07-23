#include "GICudaVoxelScene.h"
#include "GIKernels.cuh"

GICudaVoxelScene::GICudaVoxelScene()
	: dVoxGrid(nullptr)
{
	cudaMalloc(reinterpret_cast<void**>(dVoxGrid), sizeof(CVoxelGrid));
}

GICudaVoxelScene::~GICudaVoxelScene()
{
	cudaFree(dVoxGrid);
}

// Determines and Allocates the initial Page Size for the first frame
void GICudaVoxelScene::LinkOGL(GLuint aabbBuffer,
							   GLuint transformBufferID,
							   GLuint relativeTransformBufferID,
							   GLuint voxelCache,
							   GLuint voxelCacheRender)
{

}

void GICudaVoxelScene::AllocateInitialPages()
{
	// Hint Device that we will use already linked resources
}

void GICudaVoxelScene::Voxelize(const IEVector3& playerPos)
{
	// Main Call Chain Called Every Frame
	// Manages Voxel Pages


	//// Introduce Cull KC
	//VoxelObjectCull(unsigned int* gObjectIndices,
	//				unsigned int& gIndicesIndex,
	//				const CObjectAABB* gObjectAABB,
	//				const CObjectTransform* gObjTransforms,
	//				const CVoxelGrid& gGridInfo);

	//// Introduce KC
	//VoxelIntroduce(CVoxelData* gVoxelData,
	//			   const unsigned int gPageAmount,
	//			   const CVoxelPacked* gObjectVoxelCache,
	//			   const CVoxelRender* gObjectVoxelRenderCache,
	//			   const CObjectTransform& gObjTransform,
	//			   const CObjectAABB& objAABB,
	//			   const CVoxelGrid& gGridInfo);

	//// Voxel Transform KC
	//void VoxelTransform(CVoxelData* gVoxelData,
	//					CVoxelGrid& gGridInfo,
	//					const float3 newGridPos,
	//					const CObjectTransform* gObjTransformsRelative);

	////


}

GLuint GICudaVoxelScene::VoxelDataForRendering()
{
	return 0;
}