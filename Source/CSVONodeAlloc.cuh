#pragma once

#include <cuda.h>

__device__ uint32_t AtomicAllocateNode(bool& allocated, CSVONode* gNode, uint32_t* gLevelAllocator);
__device__ uint32_t TraverseNode(uint32_t& traversedLevel,
								 // SVO
								 const CSVOLevelConst* svoLevels,
								 // Node Related
								 const int3& voxelId,
								 // Constants
								 const OctreeParameters& octreeParams,
								 const uint32_t level);
__device__ uint32_t PunchThroughNode(// SVO
									 uint32_t* gLevelAllocators,
									 const uint32_t* gLevelCapacities,
									 const CSVOLevel* gSVOLevels,
									 // Node Related
									 const int3& voxelId,
									 // Constants
									 const OctreeParameters& octreeParams,
									 const uint32_t level,
									 const bool writeId);

inline __device__ uint32_t AtomicAllocateNode(bool& allocated, CSVONode* gNode, uint32_t* gLevelAllocator)
{
	// Release Configuration Optimization fucks up the code
	// Prob changes some memory i-o ordering
	// Its fixed but comment is here for future
	// Problem here was cople threads on the same warp waits eachother and
	// after some memory ordering changes by compiler responsible thread waits
	// other threads execution to be done
	// Code becomes something like this after compiler changes some memory orderings
	//
	//	while(old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE) == 0xFFFFFFFE); <-- notice semicolon
	//	 if(old == 0xFFFFFF)
	//		location = allocate();
	//	location = old;
	//	return location;
	//
	// first allocating thread will never return from that loop since 
	// its warp threads are on infinite loop (so deadlock)
	//
	// much cooler version can be warp level exchange intrinsics
	// which slightly reduces atomic pressure on the single node (on lower tree levels atleast)
	//
	// 0xFFFFFFFF means empty (non-allocated) node
	// 0xFFFFFFFE means allocation in progress
	// All other numbers are valid nodes (unless of course those are out of bounds)

	// Just take node if already allocated
	allocated = false;
	if(gNode->next < 0xFFFFFFFE) return gNode->next;
	// Try to lock the node and allocate for that node
	uint32_t old = 0xFFFFFFFE;
	while(old == 0xFFFFFFFE)
	{
		old = atomicCAS(&gNode->next, 0xFFFFFFFF, 0xFFFFFFFE);
		if(old == 0xFFFFFFFF)
		{
			// Allocate
			uint32_t location = atomicAdd(gLevelAllocator, 8);
			reinterpret_cast<volatile uint32_t&>(gNode->next) = location;
			old = location;
			allocated = true;
		}
		__threadfence();	// This is important somehow compiler changes this and makes infinite loop on same warp threads
	}
	return old;
}

inline __device__ uint32_t TraverseNode(uint32_t& traversedLevel,
										// SVO
										const CSVOLevelConst* svoLevels,
										// Node Related
										const int3& voxelId,
										// Constants
										const OctreeParameters& octreeParams,
										const uint32_t level)
{
	// Returns Node Location on That Level	
	int3 denseLevelId = CalculateParentVoxId(voxelId, octreeParams.DenseLevel, level);
	const CSVOLevelConst& denseLevel = svoLevels[octreeParams.DenseLevel];
	const CSVONode* node = denseLevel.gLevelNodes + DenseIndex(denseLevelId, octreeParams.DenseSize);

	traversedLevel = octreeParams.DenseLevel;
	while(traversedLevel < level)
	{
		const uint32_t nextNode = node->next;
		if(nextNode == 0xFFFFFFFF) break;
		traversedLevel++;
		unsigned int childId = CalculateLevelChildId(voxelId, traversedLevel, level);
		node = svoLevels[traversedLevel].gLevelNodes + nextNode + childId;
	}
	return node - svoLevels[traversedLevel].gLevelNodes;
}

inline __device__ uint32_t PunchThroughNode(// SVO
											uint32_t* gLevelAllocators,
											const uint32_t* gLevelCapacities,
											const CSVOLevel* gSVOLevels,
											// Node Related
											const int3& voxelId,
											// Constants
											const OctreeParameters& octreeParams,
											const uint32_t level,
											const bool writeId)
{
	int3 denseLevelId = CalculateParentVoxId(voxelId, octreeParams.DenseLevel, level);
	const CSVOLevel& denseLevel = gSVOLevels[octreeParams.DenseLevel];
	uint32_t denseIndex = DenseIndex(denseLevelId, octreeParams.DenseSize);
	CSVONode* node = denseLevel.gLevelNodes + denseIndex;

	// Iterate untill level (This portion's nodes should be allocated)
	for(uint32_t i = octreeParams.DenseLevel + 1; i <= level; i++)
	{
		bool allocated;
		uint32_t allocNode = AtomicAllocateNode(allocated, node, gLevelAllocators + i);

		// Specical Case for Dense-Sparse transicion
		if(i == octreeParams.DenseLevel + 1 && allocated)
		{
			denseLevel.gVoxId[denseIndex] = PackNodeId(denseLevelId, i,
													   octreeParams.CascadeCount,
													   octreeParams.CascadeBaseLevel,
													   octreeParams.MaxSVOLevel);
		}

		assert(allocNode < gLevelCapacities[i]);
		uint32_t childId = CalculateLevelChildId(voxelId, i, level);
		node = gSVOLevels[i].gLevelNodes + allocNode + childId;

		//if (allocated && writeId)
		// Force Ptr generation on nodes that are usefull
		if (writeId && i != octreeParams.MaxSVOLevel)
		{
			// Write Nodeid to ParentLoc
			uint32_t loc = allocNode + childId;
			int3 thisVoxId = CalculateParentVoxId(voxelId, i, level);
			uint32_t packedParent = PackNodeId(thisVoxId, i,
											   octreeParams.CascadeCount,
											   octreeParams.CascadeBaseLevel,
											   octreeParams.MaxSVOLevel);
			gSVOLevels[i].gVoxId[loc] = packedParent;
		}
	}
	return node - gSVOLevels[level].gLevelNodes;	
	
	//int3 denseLevelId = CalculateParentVoxId(voxelId, octreeParams.DenseLevel, level);
	//const CSVOLevel& denseLevel = gSVOLevels[octreeParams.DenseLevel];
	//CSVONode* node = denseLevel.gLevelNodes + DenseIndex (denseLevelId, octreeParams.DenseSize);

	//// Iterate untill level (This portion's nodes should be allocated)
	//#pragma unroll
	//for(uint32_t i = octreeParams.DenseLevel + 1; i <= level; i++)
	//{
	//	bool allocated;
	//	uint32_t allocNode = AtomicAllocateNode (allocated, node, gLevelAllocators + i);

	//	if(allocated && writeId)
	//	{
	//		// Write Nodeid to ParentLoc
	//		uint32_t parentLoc = node - gSVOLevels[i - 1].gLevelNodes;
	//		int3 parentVoxId = CalculateParentVoxId(voxelId, i - 1, level);
	//		uint32_t packedParent = PackNodeId (parentVoxId, i - 1,
	//											octreeParams.CascadeCount,
	//											octreeParams.CascadeBaseLevel,
	//											octreeParams.MaxSVOLevel);
	//		gSVOLevels[i - 1].gVoxId[parentLoc] = packedParent;
	//	}

	//	assert (allocNode < gLevelCapacities[i]);
	//	uint32_t childId = CalculateLevelChildId (voxelId, i, level);
	//	node = gSVOLevels[i].gLevelNodes + allocNode + childId;
	//}

	//uint32_t location = node - gSVOLevels[level].gLevelNodes;
	//// Write leaf for pointer generation (only for middle-leafs)
	////if(level != octreeParams.MaxSVOLevel)
	////if(level == octreeParams.MaxSVOLevel - 1)
	//if(level == octreeParams.MaxSVOLevel - 1)
	//{
	//	gSVOLevels[level].gVoxId[location] = PackNodeId(voxelId, level,
	//													octreeParams.CascadeCount,
	//													octreeParams.CascadeBaseLevel,
	//													octreeParams.MaxSVOLevel);
	//}
	//return location;
}