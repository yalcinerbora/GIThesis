#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"

__global__ void VoxelTransform(CVoxelData* gVoxelData,
							   CVoxelGrid& gGridInfo,
							   const float3 newGridPos,
							   const CObjectTransform* gObjTransformsRelative)
{
	unsigned int pageId = blockIdx.x % GI_BLOCK_PER_PAGE;
	unsigned int pageLocalId = threadIdx.x + pageId * blockDim.x;

	// Mem Fetch and Expand (8 byte per warp, coalesced, 0 stride)
	uint3 voxPos;
	unsigned int objectId;
	ExpandVoxelData(voxPos, objectId, gVoxelData[pageId].dGridVoxels[pageLocalId]);

	// Skip if this voxel is deleted
	// ObjectId -1 reserved for invalid Node
	if(objectId == GI_DELETED_VOXEL)
		return;

	// Generate World Position
	float4 worldPos;
	worldPos.x = gGridInfo.position.x + voxPos.x * gGridInfo.span;
	worldPos.y = gGridInfo.position.y + voxPos.y * gGridInfo.span;
	worldPos.z = gGridInfo.position.z + voxPos.z * gGridInfo.span;
	worldPos.w = 1.0f;

	// Normal Fetch (12 byte per wrap, 4 byte stride)
	float3 normal;
	normal = gVoxelData[pageId].dVoxelsRenderData[pageLocalId].normal;

	// Transformations (Fetch Irrelevant because of caching)
	// Obj count <<< Voxel Count, Rigid Objects' voxel adjacent each other
	// There is a hight chance that this transform is on the shared mem(cache) already
	// ObjectId zero reserved for static geometry
	if(objectId != GI_STATIC_GEOMETRY)
	{
		// Non static object. Transform voxel & normal
		CMatrix4x4 transform = gObjTransformsRelative[objectId - 1].transform;
		CMatrix4x4 rotation = gObjTransformsRelative[objectId - 1].rotation;
		MultMatrixSelf(worldPos, transform);
		MultMatrixSelf(normal, rotation);
	}

	// Reconstruct Voxel Indices relative to the new pos of the grid
	worldPos.x -= newGridPos.x;
	worldPos.y -= newGridPos.y;
	worldPos.z -= newGridPos.z;

	bool outOfBounds;
	outOfBounds  = (worldPos.x) < 0 || (worldPos.x > gGridInfo.dimension.x * gGridInfo.span);
	outOfBounds |= (worldPos.y) < 0 || (worldPos.x > gGridInfo.dimension.y * gGridInfo.span);
	outOfBounds |= (worldPos.z) < 0 || (worldPos.x > gGridInfo.dimension.z * gGridInfo.span);

	// Now Write
	// Discard the out of bound voxels
	// will come back into the grid
	if(!outOfBounds)
	{
		float invSpan = 1.0f / gGridInfo.span;
		voxPos.x = static_cast<unsigned int>((worldPos.x) * invSpan);
		voxPos.y = static_cast<unsigned int>((worldPos.y) * invSpan);
		voxPos.z = static_cast<unsigned int>((worldPos.z) * invSpan);

		gVoxelData[pageId].dVoxelsRenderData[pageLocalId].normal = normal;
	}
	else
	{
		// Mark this node as deleted so that "voxel introduce" kernel
		// adds a voxel to this node
		unsigned int index = atomicAdd(&gVoxelData[pageId].dEmptyElementIndex, 1);
		gVoxelData[pageId].dEmptyPos[index] = pageLocalId;
		objectId = GI_DELETED_VOXEL;
	}
	PackVoxelData(gVoxelData[pageId].dGridVoxels[pageLocalId], voxPos, objectId);
}