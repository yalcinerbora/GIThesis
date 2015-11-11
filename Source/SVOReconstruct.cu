#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CHash.cuh"
#include "CAtomicAlloc.cuh"

inline __device__ unsigned int AtomicAllocNode(CSVONode* gNode,
											   unsigned int* gSVOLock,
											   unsigned int* gSVOEmptyLoc,
											   unsigned int& gSVOLocIndex,
											   unsigned int globalId,
											   const unsigned int maxCount)
{
	CSVONode old = *gNode;

	// Already Allocated
	if(old != 0xFFFFFFFF) return old;

	// Try to lock
	unsigned int lock = atomicCAS(gSVOLock, 0xFFFFFFFF, globalId + 1);
	if(lock == 0xFFFFFFFF)
	{
		// We acquired the lock its our responsibility to allocate node
		unsigned int size = atomicSub(&gSVOLocIndex, 1); assert(size <= maxCount);
		unsigned int location = gSVOEmptyLoc[size - 1];
		atomicExch(gNode, location);

		// Zero the lock (means its written)
		unsigned int lockOut = atomicCAS(gSVOLock, globalId + 1, 0); assert(lockOut == (globalId + 1));
		return location;
	}
	else
	{
		// Wait untill locked thread 
		do
		{
			old = atomicCAS(gSVOLock, 0, 0);
			//assert(old == 0);
		}
		while(old != 0);
		return 0;
	}
}

inline __device__ CSVOColor AtomicColorAvg(CSVOColor* aColor, CSVOColor color)
{
	float4 colorAdd = UnpackSVOColor(color);
	CSVOColor assumed, old = *aColor;
	do
	{
		assumed = old;
		
		// Atomic color average upto 255 colors
		float4 colorAvg = UnpackSVOColor(assumed);
		float ratio = colorAvg.w / (colorAvg.w + 1.0f);
		if(colorAvg.w < 255.0f)
		{
			colorAvg.x = (ratio * colorAvg.x) + (colorAdd.x / (colorAvg.w + 1.0f));
			colorAvg.y = (ratio * colorAvg.y) + (colorAdd.y / (colorAvg.w + 1.0f));
			colorAvg.z = (ratio * colorAvg.z) + (colorAdd.z / (colorAvg.w + 1.0f));
			colorAvg.w += 1.0f;
		}
		old = atomicCAS(aColor, assumed, PackSVOColor(colorAvg));
	}
	while(assumed != old);
	return old;
}

__global__ void SVOReconstruct(CSVONode* gSVOSparse,
							   CSVONode* gSVODense,
							   unsigned int* gSVOLock,

							   // SVO Alloc Location Holding Data
							   unsigned int* gSVOEmptyLoc,
							   unsigned int& gSVOLocIndex,
							   const unsigned int maxSVOCount,

							   // Voxel Page Data
							   const CVoxelPage* gVoxelData,

							   const unsigned int cascadeNo,
							   const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * GI_THREAD_PER_BLOCK;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];
	if(voxelNormPos.y == 0xFFFFFFFF) return;

	// Expand Voxel
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelNormPos.x);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked,
									  cascadeNo,
									  svoConstants.numCascades);
	
	// Dense Level
	uint3 denseIndex = CalculateLevelVoxId(voxelPos, svoConstants.denseDepth,
										   svoConstants.totalDepth);

	CSVONode* gDenseLoc = gSVODense +
						  svoConstants.denseDim * svoConstants.denseDim * denseIndex.z +
						  svoConstants.denseDim * denseIndex.y +
						  denseIndex.x;

	unsigned int* gSVOLockLoc = gSVOLock + 
								svoConstants.denseDim * svoConstants.denseDim * denseIndex.z +
								svoConstants.denseDim * denseIndex.y +
								denseIndex.x;

	unsigned int location = AtomicAllocNode(gDenseLoc,
											gSVOLockLoc,
											gSVOEmptyLoc,
											gSVOLocIndex,
											globalId,
											maxSVOCount);
	// This location is the starting location of the all 8 children
	// Take offset around it
	unsigned int childOffset = CalculateLevelChildId(voxelPos,
													 svoConstants.denseDepth + 1,
													 svoConstants.totalDepth);
	location += childOffset;

	// For Each Level
	unsigned int cascadeMaxDepth = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo - 1);
	for(unsigned int i = svoConstants.denseDepth + 1; i < cascadeMaxDepth; i++)
	{
		CSVONode* gNodeLoc = gSVOSparse + location;
		
		unsigned int* gSVOLockLoc = gSVOLock + location +
									svoConstants.denseDepth * 
									svoConstants.denseDepth * 
									svoConstants.denseDepth;

		// Allocated Location
		location = AtomicAllocNode(gNodeLoc,
								   gSVOLockLoc,
								   gSVOEmptyLoc,
								   gSVOLocIndex,
								   globalId,
								   maxSVOCount);

		// Add Child Offset
		location += CalculateLevelChildId(voxelPos,
										  i + 1,
										  svoConstants.totalDepth);
	}
	// Atomically Allocated Until Leaf
}