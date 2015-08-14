#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"

__global__ void VoxelTransform(// Voxel Pages
							   CVoxelPage* gVoxelData,
							   CVoxelGrid& gGridInfo,
							   const float3& gNewGridPosition,

							   // Per Object Segment
							   ushort2** gObjectAllocLocations,

							   // Object Related
							   const unsigned int** gObjectAllocIndexLookup,
							   const CObjectTransform** gObjTransformsRelative,
							   const CVoxelRender** gVoxRenderData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = blockIdx.x / GI_BLOCK_PER_PAGE;
	unsigned int pageLocalId = globalId - (blockIdx.x / GI_BLOCK_PER_PAGE);
	unsigned int pageLocalSegmentId = globalId - (blockIdx.x / GI_BLOCK_PER_PAGE) / GI_SEGMENT_SIZE;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == 0) return;

	// Mem Fetch and Expand (8 byte per warp, coalesced, 0 stride)
	ushort2 objectId;
	uint3 voxPos;
	float3 normal;
	unsigned int voxelSpanRatio;
	unsigned int renderLoc;
	ExpandVoxelData(voxPos, normal, objectId, renderLoc, voxelSpanRatio, gVoxelData[pageId].dGridVoxels[pageLocalId]);

	// Skip if this object is not in the grid
	// Or this object is not related to the transform
	if(gObjectAllocLocations[objectId.x][gObjectAllocIndexLookup[objectId.x][objectId.y]].x == 0xFF) return;

	// Generate World Position
	// GridInfo.position is old position (previous frame's position
	float4 worldPos;
	worldPos.x = gGridInfo.position.x + voxPos.x * gGridInfo.span;
	worldPos.y = gGridInfo.position.y + voxPos.y * gGridInfo.span;
	worldPos.z = gGridInfo.position.z + voxPos.z * gGridInfo.span;
	worldPos.w = 1.0f;

	// Transformations
	// Non static object. Transform voxel & normal
	// TODO normally here we will fetch render data and look at the object type
	CMatrix4x4 transform = gObjTransformsRelative[objectId.x][objectId.y].transform;
	CMatrix4x4 rotation = gObjTransformsRelative[objectId.x][objectId.y].rotation;
	MultMatrixSelf(worldPos, transform);
	MultMatrixSelf(normal, rotation);

	// Reconstruct Voxel Indices relative to the new pos of the grid
	worldPos.x -= gNewGridPosition.x;
	worldPos.y -= gNewGridPosition.y;
	worldPos.z -= gNewGridPosition.z;

	bool outOfBounds;
	outOfBounds = (worldPos.x) < 0 || (worldPos.x > gGridInfo.dimension.x * gGridInfo.span);
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

		// Only store required stuff (less bandwidth)
		PackVoxelData(gVoxelData[pageId].dGridVoxels[pageLocalId], voxPos, normal);
	}
}