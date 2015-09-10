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
	if(objectId == 0xFFFFFFFF) return;

	bool intersects = CheckGridVoxIntersect(gGridInfo, gObjectAABB[objectId], gObjTransforms[objectId]);
	ushort2 objAlloc = gObjectAllocLocations[globalId];

	if(!intersects && objAlloc.x != 0xFFFF)
	{

		//// Non atomic dealloc
		//unsigned int linearPageId = globalId / GI_SEGMENT_PER_PAGE;
		//unsigned int linearPagelocalSegId = globalId % GI_SEGMENT_PER_PAGE;
		//gVoxelData[linearPageId].dIsSegmentOccupied[linearPagelocalSegId] = 2;
		//gObjectAllocLocations[globalId] = { 0xFFFF, 0xFFFF };

		// "Dealocate"
		unsigned int location = AtomicDeallocLoc(&(gVoxelData[objAlloc.x].dEmptySegmentStackSize)) - 1;
		if(location < GI_SEGMENT_PER_PAGE)
		{
			gVoxelData[objAlloc.x].dEmptySegmentPos[location] = objAlloc.y;
			gVoxelData[objAlloc.x].dIsSegmentOccupied[location] = 2;
			gObjectAllocLocations[globalId] = { 0xFFFF, 0xFFFF };
		}
		else
		{
			assert(false);
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
	
	// Determine Obj Id (-1 Id means this object is too small for this grid
	unsigned int objectId = gSegmentObjectId[globalId];
	if(objectId == 0xFFFFFFFF) return;

	bool intersects = CheckGridVoxIntersect(gGridInfo, gObjectAABB[objectId], gObjTransforms[objectId]);
	ushort2 objAlloc = gObjectAllocLocations[globalId];

	//intersects = false;
	// Check If this object is in
	if(intersects && objAlloc.x == 0xFFFF)
	{
		// "Allocate"
		// First Segment is responsible for sending the signal
		if(globalId == 0 || (globalId != 0 && gSegmentObjectId[globalId - 1] != objectId))
			gWriteToPages[objectId] = 1;
		

		//// Non atomic alloc
		//unsigned int linearPageId = globalId / GI_SEGMENT_PER_PAGE;
		//unsigned int linearPagelocalSegId = globalId % GI_SEGMENT_PER_PAGE;
		//gObjectAllocLocations[globalId] = 
		//{
		//	static_cast<unsigned short>(linearPageId),
		//	static_cast<unsigned short>(linearPagelocalSegId)
		//};
		//gVoxelData[linearPageId].dIsSegmentOccupied[linearPagelocalSegId] = 1;

		// Check page by page
		for(unsigned int i = 0; i < gPageAmount; i++)
		{
			unsigned int location = AtomicAllocLoc(&(gVoxelData[i].dEmptySegmentStackSize)) - 1;
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
	CVoxelObjectType objType;
	float3 normal;
	unsigned int voxelSpanRatio;
	unsigned int renderLoc;
	ExpandVoxelData(voxPos, normal, objectId, objType, voxelSpanRatio, renderLoc, gObjectVoxelCache[globalId]);

	// We need to check if this obj is not already in the page system or not
	if(gWriteToPages[objectId.x] == 1)
	{
		// Determine where to write this pixel
		unsigned int voxId = globalId - gObjectVoxStrides[objectId.x];
		unsigned int segmentId = voxId / GI_SEGMENT_SIZE;
		unsigned int segmentStart = gObjectAllocIndexLookup[objectId.x];
		ushort2 segmentLoc = gObjectAllocLocations[segmentStart + segmentId];
		unsigned int segmentLocalVoxPos = (voxId - segmentId * GI_SEGMENT_SIZE);
			
		// Even tho write signal is sent, allocator may not find position for all segments (page system full)
		if(segmentLoc.x != 0xFFFF)
		{
			// Finally Actual Voxel Write
			objectId.y = batchId;
			PackVoxelIds(gVoxelData[segmentLoc.x].dGridVoxIds[segmentLoc.y * GI_SEGMENT_SIZE + segmentLocalVoxPos],
							objectId,
							objType,
							renderLoc);
			objectId.y = segmentStart;
			objectId.y = segmentId;
		}
	}
}

__global__ void VoxelClearMarked(CVoxelPage* gVoxelData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = (globalId - pageId * GI_PAGE_SIZE);
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Check if segment is marked for clear
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == 2)
	{
		// Segment is marked for clear, clear it
		gVoxelData[pageId].dGridVoxNormPos[pageLocalId] = uint2{ 0, 0 };
		gVoxelData[pageId].dGridVoxIds[pageLocalId] = uint2{ 0xFFFFFFFF, 0xFFFFFFFF };
	}
}

__global__ void VoxelClearSignal(CVoxelPage* gVoxelData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_SEGMENT_PER_PAGE;
	unsigned int pageLocalSegmentId = (globalId - pageId * GI_SEGMENT_PER_PAGE);

	// Check if segment is marked for clear
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == 2)
		gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] = 0;
}
