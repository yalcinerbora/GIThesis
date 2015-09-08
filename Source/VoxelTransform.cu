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
							   CObjectVoxelInfo** gObjInfo,
							   CObjectAABB** gObjectAABB)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = (globalId - pageId * GI_PAGE_SIZE);
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;
//	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == 0) return;
	
	// Mem Fetch and Expand (8 byte per warp, coalesced, 0 stride)
	CVoxelObjectType objType;
	ushort2 objectId;
	unsigned int renderLoc;
	ExpandVoxelIds(renderLoc, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	// Skip if this object is not in the grid
//	if(gObjectAllocLocations[objectId.y][gObjectAllocIndexLookup[objectId.y][objectId.x]].x == 0xFFFF) return;

	// Fetch NormalPos Array
	uint3 voxPos;
	float3 normal;
	unsigned int voxelSpanRatio;
	ExpandNormalPos(voxPos, normal, voxelSpanRatio, uint2{gVoxCacheData[objectId.y][renderLoc].x, gVoxCacheData[objectId.y][renderLoc].y});
	
	// Fetch AABB, transform and span
	float4 objAABBMin = gObjectAABB[objectId.y][objectId.x].min;
	CMatrix4x4 transform = gObjTransforms[objectId.y][objectId.x].transform;
	float objSpan = gObjInfo[objectId.y][objectId.x].span;

	// Calculate Span Ratio
	float3 scaling = ExtractScaleInfo(transform);
	assert(scaling.x == scaling.y);
	assert(scaling.y == scaling.z);
	// Calculate Vox Span Ratio (if this object voxel is span higher level)
	// This operation assumes object and voxel span is related (obj is pow of two multiple of grid)
	unsigned int voxRatio = static_cast<unsigned int>(objSpan * scaling.x / gGridInfo.span);
	voxRatio--;
	voxRatio |= voxRatio >> 1;
	voxRatio |= voxRatio >> 2;
	voxRatio |= voxRatio >> 4;
	voxRatio |= voxRatio >> 8;
	voxRatio |= voxRatio >> 16;
	voxRatio++;
	voxelSpanRatio = voxRatio;

	// Generate World Position
	// start with object space position
	float4 worldPos;
	worldPos.x = objAABBMin.x + voxPos.x * objSpan;
	worldPos.y = objAABBMin.y + voxPos.y * objSpan;
	worldPos.z = objAABBMin.z + voxPos.z * objSpan;
	worldPos.w = 1.0f;

	// Transformations
	objType = CVoxelObjectType::STATIC;
	switch(objType)
	{
		case CVoxelObjectType::STATIC:
		case CVoxelObjectType::DYNAMIC:
		{
			// One Transform per voxel
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
	outOfBounds |= (worldPos.y) < 0 || (worldPos.y > gGridInfo.dimension.y * gGridInfo.span);
	outOfBounds |= (worldPos.z) < 0 || (worldPos.z > gGridInfo.dimension.z * gGridInfo.span);

	// Now Write
	// Discard the out of bound voxels
	// will come back into the grid
	if(!outOfBounds)
	{
		float invSpan = 1.0f / (gGridInfo.span);
		voxPos.x = static_cast<unsigned int>(worldPos.x * invSpan);
		voxPos.y = static_cast<unsigned int>(worldPos.y * invSpan);
		voxPos.z = static_cast<unsigned int>(worldPos.z * invSpan);

	/*	voxPos.x = 456; 
		voxPos.y = 256; 
		voxPos.z = 256;*/
		
		// Write to page
		PackVoxelNormPos(gVoxelData[pageId].dGridVoxNormPos[pageLocalId], voxPos, normal, voxelSpanRatio);
	}
}