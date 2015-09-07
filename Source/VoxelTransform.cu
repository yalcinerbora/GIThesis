#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"

__global__ void VoxelTransform(// Voxel Pages
							   CVoxelPage* gVoxelData,
							   const CVoxelGrid& gGridInfo,
							   const float3 hNewGridPosition,

							   // Per Object Segment
							   ushort2** gObjectAllocLocations,

							   // Object Related
							   unsigned int** gObjectAllocIndexLookup,
							   CObjectTransform** gObjTransforms,
							   CVoxelRender** gVoxRenderData,
							   CVoxelPacked** gVoxCacheData,
							   CObjectAABB** gObjectAABB)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = (globalId - pageId * GI_PAGE_SIZE);
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == 0) return;

	// Mem Fetch and Expand (8 byte per warp, coalesced, 0 stride)
	CVoxelObjectType objType;
	ushort2 objectId;
	unsigned int renderLoc;
	ExpandVoxelIds(renderLoc, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	// Skip if this object is not in the grid
	// Or this object is not related to the transform
	if(gObjectAllocLocations[objectId.x][gObjectAllocIndexLookup[objectId.y][objectId.x]].x == 0xFF) return;

	// Fetch NormalPos Array
	uint3 voxPos;
	float3 normal;
	unsigned int voxelSpanRatio;
	ExpandNormalPos(voxPos, normal, voxelSpanRatio, uint2 {gVoxCacheData[objectId.y][objectId.x].x, gVoxCacheData[objectId.y][objectId.x].y});
	
	// Fetch AABB
	float4 objAABBMin = gObjectAABB[objectId.y][objectId.x].min;

	// Generate World Position
	// start with object space position
	float4 worldPos;
	worldPos.x = objAABBMin.x + voxPos.x * voxelSpanRatio * gGridInfo.span;
	worldPos.y = objAABBMin.y + voxPos.y * voxelSpanRatio * gGridInfo.span;
	worldPos.z = objAABBMin.z + voxPos.z * voxelSpanRatio * gGridInfo.span;
	worldPos.w = 1.0f;

	// Transformations
	switch(objType)
	{
		case CVoxelObjectType::STATIC:
		case CVoxelObjectType::DYNAMIC:
		{
			// One Transform per voxel
			CMatrix4x4 transform = gObjTransforms[objectId.y][objectId.x].transform;
			CMatrix4x4 rotation = gObjTransforms[objectId.y][objectId.x].rotation;
								
			// Now voxel is in is world space
			MultMatrixSelf(worldPos, transform);
			MultMatrixSelf(normal, rotation);
			break;
		}
		case CVoxelObjectType::SKEL_DYNAMIC:
		{
			// TODO Implement
			break;
		}
		case CVoxelObjectType::MORPH_DYNAMIC:
		{
			// TODO Implement
			break;
		}
		default:
			assert(false);
			break;
	}

	// Reconstruct Voxel Indices relative to the new pos of the grid
	worldPos.x -= hNewGridPosition.x;
	worldPos.y -= hNewGridPosition.y;
	worldPos.z -= hNewGridPosition.z;

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
		PackVoxelNormPos(gVoxelData[pageId].dGridVoxNormPos[pageLocalId], voxPos, normal, voxelSpanRatio);
	}
}