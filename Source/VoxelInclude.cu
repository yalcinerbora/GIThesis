#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>

#include "GIKernels.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"
#include "COpenGLCommon.cuh"

// 1.0f means use max, 0.0f means use min
__device__ static const float3 aabbLookupTable[] =
{
	{ 1.0f, 1.0f, 1.0f },		// V1
	{ 0.0f, 1.0f, 1.0f },		// V2
	{ 1.0f, 0.0f, 1.0f },		// V3
	{ 1.0f, 1.0f, 0.0f },		// V4

	{ 0.0f, 0.0f, 1.0f },		// V5
	{ 0.0f, 1.0f, 0.0f },		// V6
	{ 1.0f, 0.0f, 0.0f },		// V7
	{ 0.0f, 0.0f, 0.0f }		// V8
};

__device__ unsigned int AtomicAllocLoc(unsigned int* gPos)
{
	unsigned int assumed, old = *gPos;
	do
	{
		assumed = old;
		unsigned int result = (assumed == 0) ? 0 : (assumed - 1);
		old = atomicCAS(gPos, assumed, result);
	}
	while(assumed != old);
	return old;
}

__device__ unsigned int AtomicDeallocLoc(unsigned int* gPos)
{
	unsigned int assumed, old = *gPos;
	do
	{
		assumed = old;
		unsigned int result = (assumed == GI_SEGMENT_PER_PAGE) ? GI_SEGMENT_PER_PAGE : (assumed + 1);
		old = atomicCAS(gPos, assumed, result);
	}
	while(assumed != old);
	return old;
}

__device__ void VoxelAdd(// Write Location
						 CVoxelPacked* gVoxelData,

						 // Model Space Voxel
						 const ushort2& objectId,
						 const unsigned int renderLoc,
						 const float3& normal,
						 const uint3& voxPos,
						 const uint3& voxelDim,

						 // Object Related
						 const CObjectTransform& gObjTransform,
						 const CObjectAABB& gObjAABB,
						 const CObjectVoxelInfo& objInfo,
						 const CVoxelGrid& gGridInfo)
{
	// Discard if voxel is too small
	if(voxelDim.x * voxelDim.y * voxelDim.z == 0.0f) return;

	// Generate Model Space Position from voxel
	float4 localPos;
	localPos.x = gObjAABB.min.x + voxPos.x * objInfo.span;
	localPos.y = gObjAABB.min.y + voxPos.y * objInfo.span;
	localPos.z = gObjAABB.min.z + voxPos.z * objInfo.span;
	localPos.w = 1.0f;

	// Convert it to world space
	MultMatrixSelf(localPos, gObjTransform.transform);

	unsigned int voxelSpanRatio = 0;

	// We need to construct additional voxels if this voxel spans multiple gird locations
	uint3 newVoxPos;
	for(unsigned int i = 0; i < voxelDim.x * voxelDim.y * voxelDim.z; i++)
	{
		float3 localPosSubVox;  
		localPosSubVox.x = (localPos.x) + (i % voxelDim.x) * gGridInfo.span;
		localPosSubVox.y = (localPos.y) + (i % voxelDim.y) * gGridInfo.span;
		localPosSubVox.z = (localPos.z) + (i % voxelDim.z) * gGridInfo.span;

		// For each newly introduced voxel
		// Compare world pos with grid
		// Reconstruct Voxel Indices relative to the new pos of the grid
		localPosSubVox.x -= gGridInfo.position.x;
		localPosSubVox.y -= gGridInfo.position.y;
		localPosSubVox.z -= gGridInfo.position.z;
				
		float invSpan = 1.0f / gGridInfo.span;
		newVoxPos.x = static_cast<unsigned int>((localPos.x) * invSpan);
		newVoxPos.y = static_cast<unsigned int>((localPos.y) * invSpan);
		newVoxPos.z = static_cast<unsigned int>((localPos.z) * invSpan);

		// Normal in world space
		float3 normalMult = MultMatrix(normal, gObjTransform.rotation);

		// Write Back
		PackVoxelData(gVoxelData[i], newVoxPos, normalMult, objectId, voxelSpanRatio, renderLoc);
	}
}

__device__ bool CheckGridVoxIntersect(const CVoxelGrid& gGridInfo,
									  const CObjectAABB& gObjectAABB,
									  const CObjectTransform& gObjectTransform)
{
	// Comparing two AABB (Grid Itself is an AABB)
	const CAABB gridAABB =
	{
		{ gGridInfo.position.x, gGridInfo.position.y, gGridInfo.position.z, 1.0f },
		{
			gGridInfo.position.x + gGridInfo.span * gGridInfo.dimension.x,
			gGridInfo.position.y + gGridInfo.span * gGridInfo.dimension.y,
			gGridInfo.position.z + gGridInfo.span * gGridInfo.dimension.z,
			1.0f
		},
	};

	// Construct Transformed AABB
	CAABB transformedAABB =
	{
		{ FLT_MAX, FLT_MAX, FLT_MAX, 1.0f },
		{ -FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f }
	};

	for(unsigned int i = 0; i < 8; i++)
	{
		float4 data;
		data.x = aabbLookupTable[i].x * gObjectAABB.max.x + (1.0f - aabbLookupTable[i].x) * gObjectAABB.min.x;
		data.y = aabbLookupTable[i].y * gObjectAABB.max.y + (1.0f - aabbLookupTable[i].y) * gObjectAABB.min.y;
		data.z = aabbLookupTable[i].z * gObjectAABB.max.z + (1.0f - aabbLookupTable[i].z) * gObjectAABB.min.z;
		data.w = 1.0f;

		MultMatrixSelf(data, gObjectTransform.transform);
		transformedAABB.max.x = fmax(transformedAABB.max.x, data.x);
		transformedAABB.max.y = fmax(transformedAABB.max.y, data.y);
		transformedAABB.max.z = fmax(transformedAABB.max.z, data.z);

		transformedAABB.min.x = fmin(transformedAABB.min.x, data.x);
		transformedAABB.min.y = fmin(transformedAABB.min.y, data.y);
		transformedAABB.min.z = fmin(transformedAABB.min.z, data.z);
	}
	return Intersects(gridAABB, transformedAABB);
}

__global__ void VoxelObjectDealloc(// Voxel System
								   CVoxelPage* gVoxelData,
								   const unsigned int gPageAmount,
								   const CVoxelGrid& gGridInfo,

								   // Per Object Segment Related
								   ushort2* gObjectAllocLocations,
								   const unsigned int* gSegmentObjectId,
								   const uint32_t totalSegments,

								   // Per Object Related
								   char* gWriteToPages,
								   const CObjectAABB* gObjectAABB,
								   const CObjectTransform* gObjTransforms)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Now Thread Scheme changes per objectSegment
	if(globalId >= totalSegments) return;
	
	// Determine Obj Id
	unsigned int objectId = gSegmentObjectId[globalId];
	bool intersects = CheckGridVoxIntersect(gGridInfo, gObjectAABB[objectId], gObjTransforms[objectId]);
	ushort2 objAlloc = gObjectAllocLocations[globalId];

	if(!intersects && objAlloc.x != 0xFFFF)
	{
		// "Dealocate"
		unsigned int location = AtomicDeallocLoc(&(gVoxelData[objAlloc.x].dEmptySegmentIndex)) - 1;
		if(location < GI_SEGMENT_PER_PAGE)
		{
			gVoxelData[objAlloc.x].dEmptySegmentPos[location] = objAlloc.y;
			gVoxelData[objAlloc.x].dIsSegmentOccupied[location] = 0;
			gObjectAllocLocations[globalId] = { 0xFFFF, 0xFFFF };
		}
	}
}



__global__ void VoxelObjectAlloc(// Voxel System
								 CVoxelPage* gVoxelData,
								 const unsigned int gPageAmount,
								 const CVoxelGrid& gGridInfo,

								 // Per Object Segment Related
								 ushort2* gObjectAllocLocations,
								 const unsigned int* gSegmentObjectId,
								 const uint32_t totalSegments,

								 // Per Object Related
								 char* gWriteToPages,
								 const CObjectAABB* gObjectAABB,
								 const CObjectTransform* gObjTransforms)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= totalSegments) return;
	
	// Determine Obj Id
	unsigned int objectId = gSegmentObjectId[globalId];
	bool intersects = CheckGridVoxIntersect(gGridInfo, gObjectAABB[objectId], gObjTransforms[objectId]);
	ushort2 objAlloc = gObjectAllocLocations[globalId];

	// Check If this object is in
	if(intersects && objAlloc.x == 0xFFFF)
	{
		// "Allocate"
		gWriteToPages[objectId] = 1;
		
		// Check page by page
		for(unsigned int i = 0; i < gPageAmount; i++)
		{
			//unsigned int location = atomicSub(&gVoxelData[i].dEmptySegmentIndex, 1) - 1;
			unsigned int location = AtomicAllocLoc(&(gVoxelData[i].dEmptySegmentIndex)) - 1;
			if(location < GI_SEGMENT_PER_PAGE)
			{
				gObjectAllocLocations[globalId] = 
				{
					static_cast<unsigned short>(i), 
					static_cast<unsigned short>(gVoxelData[i].dEmptySegmentPos[location])
				};
				gVoxelData[i].dIsSegmentOccupied[location] = 1;
				return;
			}
		}
	}
}

__global__ void VoxelObjectInclude(// Voxel System
								   CVoxelPage* gVoxelData,
								   const unsigned int gPageAmount,
								   const CVoxelGrid& gGridInfo,

								   // Per Object Segment Related
								   ushort2* gObjectAllocLocations,
								   const uint32_t segmentCount,
								 
								   // Per Object Related
								   char* gWriteToPages,
								   const unsigned int* gObjectVoxStrides,
								   const unsigned int* gObjectAllocIndexLookup,
								   const CObjectAABB* gObjectAABB,
								   const CObjectTransform* gObjTransforms,
								   const CObjectVoxelInfo* gObjInfo,

								   // Per Voxel Related
								   const CVoxelPacked* gObjectVoxelCache,
								   uint32_t voxCount,

								   // Batch(ObjectGroup in terms of OGL) Id
								   uint32_t batchId)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Now Thread Sceheme changes per voxel
	if(globalId >= voxCount) return;
	
	// Mem Fetch
	ushort2 objectId;
	uint3 voxPos;
	float3 normal;
	unsigned int voxelSpanRatio;
	unsigned int renderLoc;
	ExpandVoxelData(voxPos, normal, objectId, renderLoc, voxelSpanRatio, gObjectVoxelCache[globalId]);

	// We need to check if this obj is not already in the page system or not
	if(gWriteToPages[objectId.y] == 1)
	{
		//// We need to check scaling and adjust span
		//// Objects may have different voxel sizes and voxel sizes may change after scaling
		//float3 scaling = ExtractScaleInfo(gObjTransforms[objectId.y].transform);
		//assert(scaling.x == scaling.y);
		//assert(scaling.y == scaling.z);

		//unsigned int voxelDim = static_cast<unsigned int>(gObjInfo[objectId.y].span * scaling.x / gGridInfo.span);
		//unsigned int voxScale = 1; voxelDim == 0 ? 0 : 1;

		//// Determine wich voxel is this thread on that specific object
		unsigned int voxId = globalId - gObjectVoxStrides[objectId.y];
		//unsigned int segment = (voxId * voxScale) / GI_SEGMENT_SIZE;
		//unsigned int segmentStart = gObjectAllocIndexLookup[objectId.y];

		//if(segmentStart < segmentCount)
		//{
		//	ushort2 segmentLoc = gObjectAllocLocations[segmentStart + segment];

		//	//// Finally Actual Voxel Write
		//	//objectId.x = batchId;
		//	//VoxelAdd(&gVoxelData[segmentLoc.x].dGridVoxels[segmentLoc.y],
		//	//			objectId,
		//	//			renderLoc,
		//	//			normal,
		//	//			voxPos,
		//	//			voxelDim,
		//	//			gObjTransforms[objectId.y],
		//	//			gObjectAABB[objectId.y],
		//	//			gObjInfo[objectId.y],
		//	//			gGridInfo);
		//}
		
		// All done stop write signal
		// Determine a leader per object
		if(voxId == 0)
		{
			gWriteToPages[objectId.y] = 0;
		}
	}
}