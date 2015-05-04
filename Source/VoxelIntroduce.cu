#include <cuda_runtime.h>
#include <cuda.h>

#include "GIKernels.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"
#include "COpenGLCommon.cuh"

__global__ void VoxelIntroduce(CVoxelData* gVoxelData,
							   const CVoxelPacked* gObjectVoxelCache,
							   const CVoxelRender* gObjectVoxelRenderCache,
							   const CObjectTransform& gObjTransform,
							   const CObjectAABB& objAABB,
							   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int pageId = blockIdx.x % GI_BLOCK_PER_PAGE;
	//unsigned int pageLocalId = threadIdx.x + pageId * blockDim.x;

	// Mem Fetch
	unsigned int objectId;
	uint3 voxPos;
	ExpandVoxelData(voxPos, objectId, gObjectVoxelCache[globalId]);

	// Generate Model Space Position from voxel
	float4 localPos;
	localPos.x = objAABB.min.x + voxPos.x * gGridInfo.span;
	localPos.y = objAABB.min.y + voxPos.y * gGridInfo.span;
	localPos.z = objAABB.min.z + voxPos.z * gGridInfo.span;
	localPos.w = 1.0f;
	MultMatrixSelf(localPos, gObjTransform.transform);

	// Compare world pos with grid
	// Reconstruct Voxel Indices relative to the new pos of the grid
	localPos.x -= gGridInfo.position.x;
	localPos.y -= gGridInfo.position.y;
	localPos.z -= gGridInfo.position.z;

	bool outOfBounds;
	outOfBounds = (localPos.x) < 0 || (localPos.x > gGridInfo.dimension.x * gGridInfo.span);
	outOfBounds |= (localPos.y) < 0 || (localPos.x > gGridInfo.dimension.y * gGridInfo.span);
	outOfBounds |= (localPos.z) < 0 || (localPos.x > gGridInfo.dimension.z * gGridInfo.span);

	if(!outOfBounds)
	{
		float3 normal = gObjectVoxelRenderCache[globalId].normal;
		MultMatrixSelf(normal, gObjTransform.rotation);

		float invSpan = 1.0f / gGridInfo.span;
		voxPos.x = static_cast<unsigned int>((localPos.x) * invSpan);
		voxPos.y = static_cast<unsigned int>((localPos.y) * invSpan);
		voxPos.z = static_cast<unsigned int>((localPos.z) * invSpan);

		// Determine A Position
		// TODO: Optimize this (template loop unrolling)
		// page size should be adjusted to compensate that (multiples of two)
		for(unsigned int i = 0; i < gPageAmount; i++)
		{
			// Check this pages empty spaces
			unsigned int location;
			location = atomicDec(&gVoxelData[i].dEmptyElementIndex, 0xFFFFFFFF);
			if(location != 0xFFFFFFFF)
			{
				// Found a Space
				PackVoxelData(gVoxelData[i].dGridVoxels[location], voxPos, globalId + 1);
				gVoxelData[i].dVoxelsRenderData[location].normal = normal;
			}
		}
	}
}

__global__ void VoxelObjectCull(unsigned int* gObjectIndices,
								unsigned int& gIndicesIndex,
								const CObjectAABB* gObjectAABB,
								const CObjectTransform* gObjTransforms,
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
			globalGridInfo.position.x + globalGridInfo.span *globalGridInfo.dimension.x,
			globalGridInfo.position.y + globalGridInfo.span *globalGridInfo.dimension.y,
			globalGridInfo.position.z + globalGridInfo.span *globalGridInfo.dimension.z
		},
	};
	intersects = Intersects(gridAABB, gObjectAABB[globalId]);

	if(intersects)
	{
		writeIndex = atomicAdd(&gIndicesIndex, 0);
		gObjectIndices[writeIndex] = globalId;
	}
}