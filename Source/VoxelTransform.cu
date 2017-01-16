#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "GICudaAllocator.h"
#include "CHash.cuh"

#define GI_MAX_JOINT_COUNT GI_MAX_SHARED_COUNT

inline __device__ void LoadTransformData(// Shared Mem
										 CMatrix4x4* sTransformMatrices,
										 CMatrix3x3* sRotationMatrices,
										 uint8_t* sMatrixLookup,

										 // Object Transform Matrix
										 CObjectTransform** gObjTransforms,
										 CObjectTransform** gJointTransforms,
										 uint32_t** gObjTransformIds,

										 // Current Voxel Weight
										 const uchar4& voxelWeightIndex,

										 // Object Type that will be broadcasted
										 const CVoxelObjectType& objType,
										 const uint16_t& objId,
										 const uint16_t& batchId)
{
	unsigned int blockLocalId = threadIdx.x;

	// transform Id Fetched only by first warp
	unsigned int transformId = 0;
	if(blockLocalId < warpSize)
		transformId = gObjTransformIds[batchId][objId];

	// Here we will load transform and rotation matrices
	// Each thread will load 1 float. There is two 4x4 matrix
	// 32 floats will be loaded
	// Just enough for a warp to do the work
	// Load matrices (4 byte load by each thread sequential no bank conflict)
	if(blockLocalId < 16)
	{
		unsigned int columnId = blockLocalId / 4;
		unsigned int rowId = blockLocalId % 4;
		reinterpret_cast<float*>(&sTransformMatrices[0].column[columnId])[rowId] =
			reinterpret_cast<float*>(&gObjTransforms[batchId][transformId].transform.column[columnId])[rowId];
	}
	else if(blockLocalId < 28)
	{
		unsigned int rotationId = blockLocalId - 16;
		unsigned int columnId = rotationId / 3;
		unsigned int rowId = rotationId % 3;
		reinterpret_cast<float*>(&sRotationMatrices[0].column[columnId])[rowId] =
			reinterpret_cast<float*>(&gObjTransforms[batchId][transformId].rotation.column[columnId])[rowId];
	}

	// Load Joint Transforms if Skeletal Object
	if(objType == CVoxelObjectType::SKEL_DYNAMIC)
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

		// Matrix Lookup Initialize
		if(blockLocalId < GI_MAX_JOINT_COUNT)
			sMatrixLookup[blockLocalId] = 0;
		__syncthreads();

		if(voxelWeightIndex.x != 0xFF) sMatrixLookup[voxelWeightIndex.x] = 1;
		if(voxelWeightIndex.y != 0xFF) sMatrixLookup[voxelWeightIndex.y] = 1;
		if(voxelWeightIndex.z != 0xFF) sMatrixLookup[voxelWeightIndex.z] = 1;
		if(voxelWeightIndex.w != 0xFF) sMatrixLookup[voxelWeightIndex.w] = 1;
		__syncthreads();

		// Lookup Tables are Loaded
		// Theorethical 63 Matrices will be loaded
		//	Each thread will load 1 float we need 1024 threads
		unsigned int iterationCount = (GI_MAX_JOINT_COUNT * 16) / blockDim.x;
		for(unsigned int i = 0; i < iterationCount; i++)
		{
            // Transformation
            unsigned int floatCount = GI_MAX_JOINT_COUNT * 16;
            unsigned int floatId = blockLocalId + (blockDim.x * i);
			if(floatId < floatCount)
			{
                unsigned int matrixId = (floatId / 16);
                unsigned int matrixLocalFloatId = floatId % 16;
				if(sMatrixLookup[matrixId] == 1)
				{
					unsigned int column = (matrixLocalFloatId / 4) % 4;
					unsigned int row = matrixLocalFloatId % 4;

                    reinterpret_cast<float*>(&sTransformMatrices[matrixId + 1].column[column])[row] =
                        reinterpret_cast<float*>(&gJointTransforms[batchId][matrixId].transform.column[column])[row];
				}
			}
			// Rotation
            floatCount = GI_MAX_JOINT_COUNT * 9;          
			if(floatId < floatCount)
			{
                unsigned int matrixId = (floatId / 9);
                unsigned int matrixLocalFloatId = floatId % 9;
				if(sMatrixLookup[matrixId] == 1)
				{
					unsigned int column = (matrixLocalFloatId / 3) % 3;
					unsigned int row = matrixLocalFloatId % 3;
        
                    reinterpret_cast<float*>(&sRotationMatrices[matrixId + 1].column[column])[row] =
						reinterpret_cast<float*>(&gJointTransforms[batchId][matrixId].rotation.column[column])[row];
				}
			}
		}
	}

	// We write to shared mem sync between warps
	__syncthreads();
}

__global__ void VoxelTransform(// Voxel Pages
							   CVoxelPage* gVoxelData,
							   const CVoxelGrid& gGridInfo,
							   const float3 hNewGridPosition,

							   // Object Related
							   CObjectTransform** gObjTransforms,
							   CObjectTransform** gJointTransforms,
							   uint32_t** gObjTransformIds,

							   // Cache
							   CVoxelNormPos** gVoxNormPosCacheData,
							   CVoxelColor** gVoxRenderData,
							   CVoxelWeight** gVoxWeightData,

							   CObjectVoxelInfo** gObjInfo,	
							   CObjectAABB** gObjectAABB)
{
	// Cache Loading
	// Shared Memory which used for transform rendering
	__shared__ CMatrix4x4 sTransformMatrices[GI_MAX_JOINT_COUNT + 1];	// First index holds model matrix
	__shared__ CMatrix3x3 sRotationMatrices[GI_MAX_JOINT_COUNT + 1];
	__shared__ uint8_t sMatrixLookup[GI_MAX_JOINT_COUNT];

	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;
	unsigned int segmentLocalVoxId = pageLocalId % GI_SEGMENT_SIZE;

	// Get Segments Obj Information Struct
	SegmentObjData segObj = gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId];
	CVoxelObjectType objType;
	uint16_t segLoad;
	SegmentOccupation segOccup;
	ExpandSegmentPacked(objType, segOccup, segLoad, segObj.packed);

	if(segOccup == SegmentOccupation::EMPTY) return;
	assert(segOccup != SegmentOccupation::MARKED_FOR_CLEAR);
	
	// Calculate your Object VoxelId
	unsigned int cacheVoxelId = segObj.voxStride + segmentLocalVoxId;
	
	CVoxelWeight weights = {{0x00, 0x00, 0x00, 0x00}, {0xFF, 0xFF, 0xFF, 0xFF}};
	if(segmentLocalVoxId < segLoad &&
	   objType == CVoxelObjectType::SKEL_DYNAMIC)
	   weights = gVoxWeightData[segObj.batchId][cacheVoxelId];
	
	// Segment is occupied so load matrices before culling unused warps
	LoadTransformData(// Shared Mem
					  sTransformMatrices,
					  sRotationMatrices,
					  sMatrixLookup,

					  // Object Transform Matrix
					  gObjTransforms,
					  gJointTransforms,
					  gObjTransformIds,

					  // Weight Index
					  weights.weightIndex,

					  // Object Type that will be broadcasted
					  objType,
					  segObj.objId,
					  segObj.batchId);

	// Now we can cull unused threads
	if(segmentLocalVoxId >= segLoad) return;

	// Fetch NormalPos from cache
	uint3 voxPos;
	float3 normal;
	bool isMip;
	ExpandNormalPos(voxPos, normal, isMip, gVoxNormPosCacheData[segObj.batchId][cacheVoxelId]);

	// Fetch AABB min, transform and span
	float4 objAABBMin = gObjectAABB[segObj.batchId][segObj.objId].min;
	float objSpan = gObjInfo[segObj.batchId][segObj.objId].span;

	// Generate World Position
	// start with object space position
	float3 worldPos;
	worldPos.x = objAABBMin.x + voxPos.x * objSpan;
	worldPos.y = objAABBMin.y + voxPos.y * objSpan;
	worldPos.z = objAABBMin.z + voxPos.z * objSpan;

	// Transformations
	// Joint
	if(objType == CVoxelObjectType::SKEL_DYNAMIC)
	{
		float4 weightUnorm;
		weightUnorm.x = static_cast<float>(weights.weight.x) / 255.0f;
		weightUnorm.y = static_cast<float>(weights.weight.y) / 255.0f;
		weightUnorm.z = static_cast<float>(weights.weight.z) / 255.0f;
		weightUnorm.w = static_cast<float>(weights.weight.w) / 255.0f;

		//if(threadIdx.x == 0)
		//	printf("x %d, y %d, z %d, w %d\n",
		//	weights.weightIndex.x,
		//	weights.weightIndex.y,
		//	weights.weightIndex.z,
		//	weights.weightIndex.w);

		// Nyra Char Related Assert
		assert(weights.weightIndex.x <= 24);
		assert(weights.weightIndex.y <= 24);
		assert(weights.weightIndex.z <= 24);
		assert(weights.weightIndex.w <= 24);

		float3 pos = {0.0f, 0.0f, 0.0f};
		float3 p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.x + 1]);
        //float3 p = MultMatrix(worldPos, gJointTransforms[segObj.batchId][weights.weightIndex.x].transform);
        
		pos.x += weightUnorm.x * p.x;
		pos.y += weightUnorm.x * p.y;
		pos.z += weightUnorm.x * p.z;

		p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.y + 1]);
        //p = MultMatrix(worldPos, gJointTransforms[segObj.batchId][weights.weightIndex.y].transform);
		pos.x += weightUnorm.y * p.x;
		pos.y += weightUnorm.y * p.y;
		pos.z += weightUnorm.y * p.z;

		p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.z + 1]);
        //p = MultMatrix(worldPos, gJointTransforms[segObj.batchId][weights.weightIndex.z].transform);
		pos.x += weightUnorm.z * p.x;
		pos.y += weightUnorm.z * p.y;
		pos.z += weightUnorm.z * p.z;

		p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.w + 1]);
        //p = MultMatrix(worldPos, gJointTransforms[segObj.batchId][weights.weightIndex.w].transform);
		pos.x += weightUnorm.w * p.x;
		pos.y += weightUnorm.w * p.y;
		pos.z += weightUnorm.w * p.z;

		worldPos = pos;

		float3 norm = {0.0f, 0.0f, 0.0f};
		float3 n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.x + 1]);
		norm.x += weightUnorm.x * n.x;
		norm.y += weightUnorm.x * n.y;
		norm.z += weightUnorm.x * n.z;

		n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.y + 1]);
		norm.x += weightUnorm.y * n.x;
		norm.y += weightUnorm.y * n.y;
		norm.z += weightUnorm.y * n.z;

		n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.z + 1]);
		norm.x += weightUnorm.z * n.x;
		norm.y += weightUnorm.z * n.y;
		norm.z += weightUnorm.z * n.z;

		n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.w + 1]);
		norm.x += weightUnorm.w * n.x;
		norm.y += weightUnorm.w * n.y;
		norm.z += weightUnorm.w * n.z;

		normal = norm;
	}

	// Model multiplication
	MultMatrixSelf(worldPos, sTransformMatrices[0]);
	MultMatrixSelf(normal, sRotationMatrices[0]);
	//// Unoptimized Matrix Load
	//CMatrix4x4 transform = gObjTransforms[segObj.batchId][gObjTransformIds[segObj.batchId][segObj.objId]].transform;
	//CMatrix4x4 rotation = gObjTransforms[segObj.batchId][gObjTransformIds[segObj.batchId][segObj.objId]].transform;
	//MultMatrixSelf(worldPos, transform);
	//MultMatrixSelf(normal, rotation);

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

	// Voxel Space
	float invSpan = 1.0f / gGridInfo.span;
	voxPos.x = static_cast<unsigned int>(worldPos.x * invSpan);
	voxPos.y = static_cast<unsigned int>(worldPos.y * invSpan);
	voxPos.z = static_cast<unsigned int>(worldPos.z * invSpan);	

	// Calculate VoxelWeights
	float3 volumeWeight;
	volumeWeight.x = worldPos.x * invSpan;
	volumeWeight.y = worldPos.y * invSpan;
	volumeWeight.z = worldPos.z * invSpan;
	
	volumeWeight.x = volumeWeight.x - static_cast<float>(voxPos.x);
	volumeWeight.y = volumeWeight.y - static_cast<float>(voxPos.y);
	volumeWeight.z = volumeWeight.z - static_cast<float>(voxPos.z);

	//volumeWeight.x = 1.0f;
	//volumeWeight.y = 1.0f;
	//volumeWeight.z = 1.0f;

	uint3 neigbourBits;
	neigbourBits.x = (volumeWeight.x > 0) ? 1 : 0;
	neigbourBits.y = (volumeWeight.y > 0) ? 1 : 0;
	neigbourBits.z = (volumeWeight.z > 0) ? 1 : 0;
		
	// Outer Bound Check
	outOfBounds |= (voxPos.x >= gGridInfo.dimension.x);
	outOfBounds |= (voxPos.y >= gGridInfo.dimension.y);
	outOfBounds |= (voxPos.z >= gGridInfo.dimension.z);

	// Now Write
	// Discard the out of bound voxels
	//outOfBounds = false;
	if(!outOfBounds)
	{
		// Write to page
		uint2 packedVoxNormPos;

		PackVoxelNormPos(packedVoxNormPos, voxPos, normal, isMip);
		gVoxelData[pageId].dGridVoxPos[pageLocalId] = packedVoxNormPos.x;
		gVoxelData[pageId].dGridVoxNorm[pageLocalId] = packedVoxNormPos.y;
				
		gVoxelData[pageId].dGridVoxOccupancy[pageLocalId] = PackOccupancy(neigbourBits, volumeWeight);
	}
	else
	{
		gVoxelData[pageId].dGridVoxPos[pageLocalId] = 0xFFFFFFFF;
		gVoxelData[pageId].dGridVoxNorm[pageLocalId] = 0xFFFFFFFF;
	}
}