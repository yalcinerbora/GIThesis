#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"
#include "CHash.cuh"

__device__ void LoadTransformData(// Shared Mem
								  unsigned int* sHashIndex,
								  CMatrix4x4* sTransformMatrices,
								  CMatrix4x4* sRotationMatrices,

								  // Object Transform Matrix
								  CObjectTransform** gObjTransforms,

								  // Object Type that will be broadcasted
								  const CVoxelObjectType& objType,
								  const ushort2& objectId)

{
	__shared__ CVoxelObjectType sObjType;
	unsigned int blockLocalId = threadIdx.x;
	
	// Broadcast objType
	if(blockLocalId == 0)
	{
		sObjType = objType;
	}
	__syncthreads();
	
	// Init Index Cache if this object type is morph dynamic or skeleton dynamic
	if((sObjType != CVoxelObjectType::STATIC || 
		sObjType != CVoxelObjectType::DYNAMIC) &&
		blockLocalId < GI_MAX_SHARED_COUNT_PRIME)
	{
		sHashIndex[blockLocalId] = 0;
	}
	__syncthreads();


	// Each Voxel Type Has Different Deformation(Animation)
	switch(objType)
	{
		case CVoxelObjectType::STATIC:
		case CVoxelObjectType::DYNAMIC:
		{
			// Static or Dynamic Objects have single transformation matrix to animate
			// they also have rotation only matrix for normal manipulation

			// Here we will load transform and rotation matrices
			// Each thread will load 1 float. There is two 4x4 matrix
			// 32 floats will be loaded
			// Just enough for a warp to do the work
			// Because of that we will broadcast obj id using the first warp
			// Pack objId to int
			unsigned int objIdShuffle;
			objIdShuffle = static_cast<unsigned int>(objectId.y) << 16;
			objIdShuffle |= static_cast<unsigned int>(objectId.x);

			// Broadcast
			objIdShuffle = __shfl(objIdShuffle, 0);
			
			// Unpack broadcasted objId to ushort2
			ushort2 objIdAfterShuffle;
			objIdAfterShuffle.x = (objIdShuffle & 0x0000FFFF);
			objIdAfterShuffle.y = (objIdShuffle & 0xFFFF0000) >> 16;
		
			// Load matrices
			if(blockLocalId < 16)
			//if(blockLocalId < 4)
			{
				if(objIdAfterShuffle.x == 244)
				{
					assert(blockLocalId != 15);
				}

				reinterpret_cast<float*>(&sTransformMatrices[0].column[blockLocalId / 4])[blockLocalId % 4] =
					reinterpret_cast<float*>(&gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].transform.column[blockLocalId / 4])[blockLocalId % 4];
				//sTransformMatrices[0].column[blockLocalId % 4] = gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].transform.column[blockLocalId % 4];
			}
			else if(blockLocalId < 32)
			//else if(blockLocalId < 8)
			{
				blockLocalId -= 16;
				reinterpret_cast<float*>(&sRotationMatrices[0].column[blockLocalId / 4])[blockLocalId % 4] =
					reinterpret_cast<float*>(&gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].rotation.column[blockLocalId / 4])[blockLocalId % 4];
				//sRotationMatrices[0].column[blockLocalId % 4] = gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].rotation.column[blockLocalId % 4];
			}
			
			//if(blockLocalId == 0)
			//{
			//	sTransformMatrices[0] = gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].transform;
			//	//{{
			//	//	{0.19f, 0.0f, 0.0f, 0.0f},
			//	//	{0.0f, 0.19f, 0.0f, 0.0f},
			//	//	{0.0f, 0.0f, 0.19f, 0.0f},
			//	//	{0.0f, 0.0f, 0.0f, 0.19f},
			//	//}};
			//}
			//else if(blockLocalId == 1)
			//{
			//	sRotationMatrices[0] = gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].rotation;
			//	//{{
			//	//	{1.0f, 0.0f, 0.0f, 0.0f},
			//	//	{0.0f, 1.0f, 0.0f, 0.0f},
			//	//	{0.0f, 0.0f, 1.0f, 0.0f},
			//	//	{0.0f, 0.0f, 0.0f, 1.0f},
			//	//}};
			//}
			break;
		}
		case CVoxelObjectType::SKEL_DYNAMIC:
		{
			// TODO Implement
			//
			break;
		}
		case CVoxelObjectType::MORPH_DYNAMIC:
		{
			// TODO Implement
			//
			break;
		}
		default:
			assert(false);
			break;
	}
	// We write to shared mem sync between warps
	__syncthreads();
}

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
	// CacheLoading
	// Shared Memory which used for transform rendering
	__shared__ unsigned int sHashIndex[GI_MAX_SHARED_COUNT_PRIME];
	__shared__ CMatrix4x4 sTransformMatrices[GI_MAX_SHARED_COUNT_PRIME];
	__shared__ CMatrix4x4 sRotationMatrices[GI_MAX_SHARED_COUNT_PRIME];
		
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);
	
	// Fetch this voxel's id chunk from page
	CVoxelObjectType objType;
	ushort2 objectId;
	unsigned int renderLoc;

	CVoxelIds voxIdPacked = gVoxelData[pageId].dGridVoxIds[pageLocalId];
	ExpandVoxelIds(renderLoc, objectId, objType, voxIdPacked);

	// Segment is occupied so load matrices before culling unused warps
	LoadTransformData(// Shared Mem
					  sHashIndex,
					  sTransformMatrices,
					  sRotationMatrices,

					  // Object Transform Matrix
					  gObjTransforms,

					  // Object Type that will be broadcasted
					  objType,
					  objectId);

	// Cull unused warps
	if(voxIdPacked.x == 0xFFFFFFFF && voxIdPacked.y == 0xFFFFFFFF) return;

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
			// Entire block will do the assertion
			if(objectId.x != 7 &&
			   objectId.x != 16 )
			{
				//assert(fabs(sTransformMatrices[0].column[0].x - 0.18954435f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[0].y - 0.0f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[0].z - 0.0f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[0].w - 0.0f) < 0.001f);

				//assert(fabs(sTransformMatrices[0].column[1].x - 0.0f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[1].y - 0.18954435f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[1].z - 0.0f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[1].w - 0.0f) < 0.001f);

				//assert(fabs(sTransformMatrices[0].column[2].x - 0.0f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[2].y - 0.0f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[2].z - 0.18954435f) < 0.001f);
				//assert(fabs(sTransformMatrices[0].column[2].w - 0.0f) < 0.001f);

				//assert(sTransformMatrices[0].column[3].x == 0.0f);
				//assert(sTransformMatrices[0].column[3].y == 0.0f);
				//assert(sTransformMatrices[0].column[3].z == 0.0f);
				//assert(sTransformMatrices[0].column[3].w == 1.0f);


				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[0].x == 0.18954435f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[0].y == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[0].z == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[0].w == 0.0f);

				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[1].x == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[1].y == 0.18954435f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[1].z == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[1].w == 0.0f);

				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[2].x == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[2].y == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[2].z == 0.18954435f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[2].w == 0.0f);

				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[3].x == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[3].y == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[3].z == 0.0f);
				//assert(gObjTransforms[objectId.y][objectId.x].transform.column[3].w == 1.0f);
			}
			// Now voxel is in is world space
			MultMatrixSelf(worldPos, sTransformMatrices[0]);
			MultMatrixSelf(normal, sRotationMatrices[0]);
			break;
		}
		case CVoxelObjectType::SKEL_DYNAMIC:
		{
			// TODO Implement
			//
			break;
		}
		case CVoxelObjectType::MORPH_DYNAMIC:
		{
			// TODO Implement
			//
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