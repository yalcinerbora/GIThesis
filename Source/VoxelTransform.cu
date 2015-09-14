#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"
#include "CHash.cuh"

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
	unsigned int blockLocalId = threadIdx.x;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);
	
	 // CacheLoading
	 // Each Block has half a segment 
	 // Segment has an object
	 // Object worst case contains multiple of matrices
	__shared__ unsigned int sHashIndex[GI_MAX_SHARED_COUNT_PRIME];
	__shared__ CMatrix4x4 sTransformMatrices[GI_MAX_SHARED_COUNT_PRIME];
	__shared__ CMatrix4x4 sRotationMatrices[GI_MAX_SHARED_COUNT_PRIME];

	// Fetch this voxel's id chunk from page
	// Skip if its invalid
	CVoxelIds voxIdPacked = gVoxelData[pageId].dGridVoxIds[pageLocalId];
	if(voxIdPacked.x == 0xFFFFFFFF && voxIdPacked.y == 0xFFFFFFFF) return;

	CVoxelObjectType objType;
	ushort2 objectId;
	unsigned int renderLoc;
	ExpandVoxelIds(renderLoc, objectId, objType, voxIdPacked);

	// Init Index Cache if this object type is morph dynamic of skeleton dynamic
	if((objType != CVoxelObjectType::STATIC || objType != CVoxelObjectType::DYNAMIC) &&
		blockLocalId < GI_MAX_SHARED_COUNT_PRIME)
	{
		sHashIndex[blockLocalId] = 0;
	}
	__syncthreads();
	
	// Fetch NormalPos from cache
	uint3 voxPos;
	float3 normal;
	unsigned int voxelSpanRatio;
	ExpandNormalPos(voxPos, normal, voxelSpanRatio, uint2{gVoxCacheData[objectId.y][renderLoc].x, gVoxCacheData[objectId.y][renderLoc].y});

	// Fetch AABB min, transform and span
	float4 objAABBMin = gObjectAABB[objectId.y][objectId.x].min;
	float objSpan = gObjInfo[objectId.y][objectId.x].span;

	// Generate World Position
	// start with object space position
	float4 worldPos;
	worldPos.x = objAABBMin.x + voxPos.x * objSpan;
	worldPos.y = objAABBMin.y + voxPos.y * objSpan;
	worldPos.z = objAABBMin.z + voxPos.z * objSpan;
	worldPos.w = 1.0f;

	// Transformations
	switch(objType)
	{
		case CVoxelObjectType::STATIC:
		case CVoxelObjectType::DYNAMIC:
		{
			// Load matrices
			if(blockLocalId < 16)
			{
				// Load Transform
				reinterpret_cast<float*>(&sTransformMatrices[0].column[blockLocalId / 4])[blockLocalId % 4] = 
					reinterpret_cast<float*>(&gObjTransforms[objectId.y][objectId.x].transform.column[blockLocalId / 4])[blockLocalId % 4];
			}
			else if(blockLocalId < 32)
			{
				reinterpret_cast<float*>(&sRotationMatrices[0].column[blockLocalId / 4])[blockLocalId % 4] =
					reinterpret_cast<float*>(&gObjTransforms[objectId.y][objectId.x].rotation.column[blockLocalId / 4])[blockLocalId % 4];
			}
			__syncthreads();

			// One Transform per voxel
			// 
			//CMatrix4x4 rotation = gObjTransforms[objectId.y][objectId.x].rotation;
			CMatrix4x4 rotation = sRotationMatrices[0];
			//{{
			//	{1.0f, 0.0f, 0.0f, 0.0f},
			//	{0.0f, 1.0f, 0.0f, 0.0f},
			//	{0.0f, 0.0f, 1.0f, 0.0f},
			//	{0.0f, 0.0f, 0.0f, 1.0f},
			//}};
			//CMatrix4x4 transform = gObjTransforms[objectId.y][objectId.x].transform;
			CMatrix4x4 transform = sTransformMatrices[0];
			//{{
			//	{0.19f, 0.0f, 0.0f, 0.0f},
			//	{0.0f, 0.19f, 0.0f, 0.0f},
			//	{0.0f, 0.0f, 0.19f, 0.0f},
			//	{0.0f, 0.0f, 0.0f, 0.19f},
			//}};
		
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
	outOfBounds = (worldPos.x < 0.0f) || (worldPos.x > gGridInfo.dimension.x * gGridInfo.span);
	outOfBounds |= (worldPos.y < 0.0f) || (worldPos.y > gGridInfo.dimension.y * gGridInfo.span);
	outOfBounds |= (worldPos.z < 0.0f) || (worldPos.z > gGridInfo.dimension.z * gGridInfo.span);

	// Now Write
	// Discard the out of bound voxels
	// will come back into the grid
	if(!outOfBounds)
	{
		float invSpan = 1.0f / gGridInfo.span;
		voxPos.x = static_cast<unsigned int>(worldPos.x * invSpan);
		voxPos.y = static_cast<unsigned int>(worldPos.y * invSpan);
		voxPos.z = static_cast<unsigned int>(worldPos.z * invSpan);

		// Write to page
		PackVoxelNormPos(gVoxelData[pageId].dGridVoxNormPos[pageLocalId], voxPos, normal, voxelSpanRatio);
	}
	else
	{
		gVoxelData[pageId].dGridVoxNormPos[pageLocalId] = uint2{0, 0};
	}
}