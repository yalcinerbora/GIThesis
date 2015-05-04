#include <cuda_runtime.h>
#include <cuda.h>

#include "GIKernels.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"
#include "COpenGLCommon.cuh"

__global__ void VoxelIntroduce(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,
							   unsigned int* gEmptyMarkArray,
							   unsigned int& gEmptyMarkIndex,
							   const CVoxelPacked* gObjectVoxelCache,
							   const CVoxelRender* gObjectVoxelRenderCache,
							   const CObjectTransformOGL& gObjTransform,
							   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;


	uint3 voxPos;
	unsigned int objectId;
	float3 normal;
	bool outOfBounds;

	// Transform Voxel
	ExpandVoxelData(voxPos, objectId, gVoxelData[globalId]);


	// Find Voxel pos in grid
	// if voxel is inside
		// Add voxel to the grid
	//





}

__global__ void VoxelObjectCull(unsigned int* gObjectIndices,
								unsigned int& gIndicesIndex,
								const CObjectAABBOGL* gObjectAABB,
								const CObjectTransformOGL* gObjTransforms,
								const CVoxelGrid& globalGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int writeIndex;
	bool intersects = false;

	// Comparing two AABB (Grid Itself is an AABB)
	CAABB gridAABB =
	{
		{globalGridInfo.position.x, globalGridInfo.position.y, globalGridInfo.position.z},
		{
			globalGridInfo.position.x + globalGridInfo.position.w *globalGridInfo.dimension.x,
			globalGridInfo.position.y + globalGridInfo.position.w *globalGridInfo.dimension.y,
			globalGridInfo.position.z + globalGridInfo.position.w *globalGridInfo.dimension.z
		},
	};
	intersects = Intersects(gridAABB, gObjectAABB[globalId]);

	if(intersects)
	{
		writeIndex = atomicAdd(&gIndicesIndex, 0);
		gObjectIndices[writeIndex] = globalId;
	}
}