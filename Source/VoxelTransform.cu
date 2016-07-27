#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"
#include "CHash.cuh"

inline __device__ void LoadTransformData(// Shared Mem
										 CMatrix4x4* sTransformMatrices,
										 CMatrix3x3* sRotationMatrices,
										 uint8_t* sMatrixLookup,

										 // Object Transform Matrix
										 CObjectTransform** gObjTransforms,
										 uint32_t** gObjTransformIds,

										 // Object Type that will be broadcasted
										 const CVoxelObjectType& objType,
										 const ushort2& objectId)
{
	__shared__ CVoxelObjectType sObjType;
	unsigned int blockLocalId = threadIdx.x;
	unsigned int transformId;
	
	// Broadcast objType
	if(blockLocalId == 0)
	{
		transformId = gObjTransformIds[objectId.y][objectId.x];
		sObjType = objType;
	}
	__syncthreads();
	
	// Here we will load transform and rotation matrices
	// Each thread will load 1 float. There is two 4x4 matrix
	// 32 floats will be loaded
	// Just enough for a warp to do the work
	// Because of that we will broadcast obj id using the first warp
	// Pack objId to int
	unsigned int objIdShuffle;
	objIdShuffle = static_cast<unsigned int>(objectId.y) << 16;
	objIdShuffle |= static_cast<unsigned int>(transformId);

	// Broadcast
	#if __CUDA_ARCH__ >= 300
		objIdShuffle = __shfl(objIdShuffle, 0);
	#else
		__shared__ unsigned int sObjId;
		if(blockLocalId == 0) sObjId = objIdShuffle;
		objIdShuffle = sObjId;
	#endif

	// Unpack broadcasted objId to ushort2
	ushort2 objIdAfterShuffle;
	objIdAfterShuffle.x = (objIdShuffle & 0x0000FFFF);
	objIdAfterShuffle.y = (objIdShuffle & 0xFFFF0000) >> 16;

	// Load matrices (4 byte load by each thread sequential no bank conflict)
	if(blockLocalId < 16)
	{
		reinterpret_cast<float*>(&sTransformMatrices[0].column[blockLocalId / 4])[blockLocalId % 4] =
			reinterpret_cast<float*>(&gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].transform.column[blockLocalId / 4])[blockLocalId % 4];
	}
	else if(blockLocalId < 28)
	{
		blockLocalId -= 16;
		reinterpret_cast<float*>(&sRotationMatrices[0].column[blockLocalId / 3])[blockLocalId % 3] =
			reinterpret_cast<float*>(&gObjTransforms[objIdAfterShuffle.y][objIdAfterShuffle.x].rotation.column[blockLocalId / 3])[blockLocalId % 3];
	}

	// Load Joint Transforms if Skeletal Object
	if(sObjType == CVoxelObjectType::SKEL_DYNAMIC)
	{
		// All valid objects will request matrix load
		// then entire block will try to load it
		// Max skeleton bone count is 64
		// Worst case 64 * 16 = 1024 float will be loaded to sMem
		// Some blocks will load twice
		// However its extremely rare (even impossible case)
		// In a realistic scenario (and if a segment holds adjacent voxels)
		// And if max bone influence per vertex is around 4 
		// there should be at most 8
		

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
							   uint32_t** gObjTransformIds,
							   CVoxelNormPos** gVoxNormPosCacheData,
							   CVoxelColor** gVoxRenderData,
							   //CVoxelWeight** gVoxelWeights,
							   CObjectVoxelInfo** gObjInfo,	
							   CObjectAABB** gObjectAABB)
{
	// CacheLoading
	// Shared Memory which used for transform rendering
	__shared__ unsigned int sBlockBail;
	__shared__ CMatrix4x4 sTransformMatrices[GI_MAX_SHARED_COUNT];
	__shared__ CMatrix3x3 sRotationMatrices[GI_MAX_SHARED_COUNT];
	__shared__ uint8_t sMatrixLookup[GI_MAX_SHARED_COUNT];

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

	// Check if this subsegment contains any voxels
	bool localBail = static_cast<unsigned int>(voxIdPacked.x == 0xFFFFFFFF && voxIdPacked.y == 0xFFFFFFFF);
	if(threadIdx.x == 0) sBlockBail = localBail;
	__syncthreads();
	if(sBlockBail) return;

	// Segment is occupied so load matrices before culling unused warps
	LoadTransformData(// Shared Mem
					  sTransformMatrices,
					  sRotationMatrices,
					  sMatrixLookup,

					  // Object Transform Matrix
					  gObjTransforms,
					  gObjTransformIds,

					  // Object Type that will be broadcasted
					  objType,
					  objectId);

	// Cull unused warps
	if(localBail) return;

	// Fetch NormalPos from cache
	uint3 voxPos;
	float3 normal;
	float4 normalWithOcc;
	bool isMip;
	ExpandNormalPos(voxPos, normalWithOcc, isMip, gVoxNormPosCacheData[objectId.y][renderLoc]);
	normal = {normalWithOcc.x, normalWithOcc.y, normalWithOcc.z};

	// Fetch AABB min, transform and span
	float4 objAABBMin = gObjectAABB[objectId.y][objectId.x].min;
	float objSpan = gObjInfo[objectId.y][objectId.x].span;

	// Generate World Position
	// start with object space position
	float3 worldPos;
	worldPos.x = objAABBMin.x + voxPos.x * objSpan;
	worldPos.y = objAABBMin.y + voxPos.y * objSpan;
	worldPos.z = objAABBMin.z + voxPos.z * objSpan;

	// Transformations
	// Model multiplication
	MultMatrixSelf(worldPos, sTransformMatrices[0]);
	MultMatrixSelf(normal, sRotationMatrices[0]);
	//// Unoptimized Matrix Load
	//CMatrix4x4 transform = gObjTransforms[objectId.y][objectId.x].transform;
	//CMatrix4x4 rotation = gObjTransforms[objectId.y][objectId.x].transform;
	//MultMatrixSelf(worldPos, transform);
	//MultMatrixSelf(normal, rotation);

	if(objType == CVoxelObjectType::SKEL_DYNAMIC)
	{
		// TODO: Apply joint transforms to the voxel
	}

	switch(objType)
	{
		case CVoxelObjectType::STATIC:
		case CVoxelObjectType::DYNAMIC:
		{
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
			// Not on Thesis Anymore
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
	outOfBounds = (worldPos.x < 0.0f) || (worldPos.x >= gGridInfo.dimension.x * gGridInfo.span);
	outOfBounds |= (worldPos.y < 0.0f) || (worldPos.y >= gGridInfo.dimension.y * gGridInfo.span);
	outOfBounds |= (worldPos.z < 0.0f) || (worldPos.z >= gGridInfo.dimension.z * gGridInfo.span);

	// If its mip dont update inner cascade
	bool inInnerCascade = false;
	if(isMip)
	{
		inInnerCascade = (worldPos.x > gGridInfo.dimension.x * gGridInfo.span * 0.25f) &&
						 (worldPos.x < gGridInfo.dimension.x * gGridInfo.span * 0.75f);
		inInnerCascade &= (worldPos.y > gGridInfo.dimension.y * gGridInfo.span * 0.25f) &&
						  (worldPos.y < gGridInfo.dimension.y * gGridInfo.span * 0.75f);
		inInnerCascade &= (worldPos.z > gGridInfo.dimension.z * gGridInfo.span * 0.25f) &&
						  (worldPos.z < gGridInfo.dimension.z * gGridInfo.span * 0.75f);
	}
	outOfBounds |= inInnerCascade;

	// Now Write
	// Discard the out of bound voxels
	if(!outOfBounds)
	{
		float invSpan = 1.0f / gGridInfo.span;
		voxPos.x = static_cast<unsigned int>(worldPos.x * invSpan + 0.5f);
		voxPos.y = static_cast<unsigned int>(worldPos.y * invSpan + 0.5f);
		voxPos.z = static_cast<unsigned int>(worldPos.z * invSpan + 0.5f);

		// Write to page
		uint2 packedVoxNormPos;
		normalWithOcc = {normal.x, normal.y, normal.z, 0.0f};
		PackVoxelNormPos(packedVoxNormPos, voxPos, normalWithOcc, isMip);
		gVoxelData[pageId].dGridVoxPos[pageLocalId] = packedVoxNormPos.x;
		gVoxelData[pageId].dGridVoxNorm[pageLocalId] = packedVoxNormPos.y;
	}
	else
	{
		gVoxelData[pageId].dGridVoxPos[pageLocalId] = 0xFFFFFFFF;
		gVoxelData[pageId].dGridVoxNorm[pageLocalId] = 0xFFFFFFFF;
	}
}