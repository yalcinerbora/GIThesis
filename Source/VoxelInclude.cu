#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>

#include "GIKernels.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"
#include "COpenGLCommon.cuh"

__device__ unsigned int AtomicAlloc(unsigned int* gStackSize)
{
	unsigned int assumed, old = *gStackSize;
	do
	{
		assumed = old;
		unsigned int result = (assumed == 0) ? 0 : (assumed - 1);
		old = atomicCAS(gStackSize, assumed, result);
	}
	while(assumed != old);
	return old;
}

__device__ unsigned int AtomicDealloc(unsigned int* gStackSize)
{
	unsigned int assumed, old = *gStackSize;
	do
	{
		assumed = old;
		unsigned int result = (assumed == GI_SEGMENT_PER_PAGE) ? GI_SEGMENT_PER_PAGE : (assumed + 1);
		old = atomicCAS(gStackSize, assumed, result);
	}
	while(assumed != old);
	return old;
}

__global__ void VoxelObjectDealloc(// Voxel System
								   CVoxelPage* gVoxelData,
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
	
	//intersects = true;
	if(!intersects && objAlloc.x != 0xFFFF)
	{

		// Non atomic dealloc
		unsigned int linearPageId = globalId / GI_SEGMENT_PER_PAGE;
		unsigned int linearPagelocalSegId = globalId % GI_SEGMENT_PER_PAGE;
		gVoxelData[linearPageId].dIsSegmentOccupied[linearPagelocalSegId] = SegmentOccupation::MARKED_FOR_CLEAR;
		gObjectAllocLocations[globalId] = ushort2{0xFFFF, 0xFFFF};

		//// "Dealocate"
		//unsigned int size = AtomicDealloc(&(gVoxelData[objAlloc.x].dEmptySegmentStackSize));
		//if(size != GI_SEGMENT_PER_PAGE)
		//{
		//	unsigned int location = size;
		//	gVoxelData[objAlloc.x].dEmptySegmentPos[location] = objAlloc.y;
		//	gVoxelData[objAlloc.x].dIsSegmentOccupied[location] = SegmentOccupation::MARKED_FOR_CLEAR;
		//	gObjectAllocLocations[globalId] = ushort2{0xFFFF, 0xFFFF};
		//}
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

	// Check If this object is in
	if(intersects && objAlloc.x == 0xFFFF)
	{
		// "Allocate"
		// First Segment is responsible for sending the signal
		if(globalId == 0 || (globalId != 0 && gSegmentObjectId[globalId - 1] != objectId))
			gWriteToPages[objectId] = 1;
		

		// Non atomic alloc
		unsigned int linearPageId = globalId / GI_SEGMENT_PER_PAGE;
		unsigned int linearPagelocalSegId = globalId % GI_SEGMENT_PER_PAGE;
		gObjectAllocLocations[globalId] = ushort2
		{
			static_cast<unsigned short>(linearPageId),
			static_cast<unsigned short>(linearPagelocalSegId)
		};
		gVoxelData[linearPageId].dIsSegmentOccupied[linearPagelocalSegId] = SegmentOccupation::OCCUPIED;

		// Check page by page
		//for(unsigned int i = 0; i < gPageAmount; i++)
		//{
		//	unsigned int size = AtomicAlloc(&(gVoxelData[i].dEmptySegmentStackSize));
		//	if(size != 0)
		//	{
		//		unsigned int location = size - 1;
		//		assert(gVoxelData[i].dIsSegmentOccupied[location] == SegmentOccupation::EMPTY);
		//		gObjectAllocLocations[globalId] = ushort2
		//		{
		//			static_cast<unsigned short>(i),
		//			static_cast<unsigned short>(gVoxelData[i].dEmptySegmentPos[location])
		//		};
		//		gVoxelData[i].dIsSegmentOccupied[location] = SegmentOccupation::OCCUPIED;
		//		return;
		//	}
		//}
	}
}

__global__ void VoxelObjectInclude(// Voxel System
								   CVoxelPage* gVoxelData,
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
	//if(objectId.x < 20)
	if(gWriteToPages[objectId.x] == 1)
	{
		// Determine where to write this pixel
		unsigned int objectLocalVoxId = globalId - gObjectVoxStrides[objectId.x];
		unsigned int segmentId = objectLocalVoxId / GI_SEGMENT_SIZE;
		unsigned int segmentStart = gObjectAllocIndexLookup[objectId.x];
		ushort2 segmentLoc = gObjectAllocLocations[segmentStart + segmentId];
		unsigned int segmentLocalVoxPos = objectLocalVoxId % GI_SEGMENT_SIZE;
			
		// Even tho write signal is sent, allocator may not find position for all segments (page system full)
		if(segmentLoc.x != 0xFFFF)
		{
			// Finally Actual Voxel Write
			objectId.y = batchId;
			PackVoxelIds(gVoxelData[segmentLoc.x].dGridVoxIds[segmentLoc.y * GI_SEGMENT_SIZE + segmentLocalVoxPos],
							objectId,
							objType,
							renderLoc);
		}
	}
}

__global__ void VoxelClearMarked(CVoxelPage* gVoxelData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Check if segment is marked for clear
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR)
	{
		// Segment is marked for clear, clear it
		gVoxelData[pageId].dGridVoxNormPos[pageLocalId] = uint2{ 0, 0 };
		gVoxelData[pageId].dGridVoxIds[pageLocalId] = uint2{ 0xFFFFFFFF, 0xFFFFFFFF };
	}
}

__global__ void VoxelClearSignal(CVoxelPage* gVoxelData,
								 const uint32_t numPages)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_SEGMENT_PER_PAGE;
	unsigned int pageLocalSegmentId = globalId % GI_SEGMENT_PER_PAGE;

	// Check if segment is marked for clear
	if(globalId >= numPages * GI_SEGMENT_PER_PAGE) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR)
		gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] = SegmentOccupation::EMPTY;
}