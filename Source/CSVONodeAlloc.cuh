#pragma once

#include <cuda.h>

//// No Negative Dimension Expansion (Best case)
//__constant__ static const char3 voxLookup8[8] =
//{
//	{0, 0, 0},
//	{1, 0, 0},
//	{0, 1, 0},
//	{1, 1, 0},
//
//	{0, 0, 1},
//	{1, 0, 1},
//	{0, 1, 1},
//	{1, 1, 1}
//};

//// Single Negative Dimension Expansion
//__constant__ static const char3 voxLookup12[12] =
//{
//	{-1, 0, 0},
//	{ 0, 0, 0},
//	{ 1, 0, 0},
//
//	{-1, 1, 0},
//	{ 0, 1, 0},
//	{ 1, 1, 0},
//
//	{-1, 0, 1},
//	{ 0, 0, 1},
//	{ 1, 0, 1},
//
//	{-1, 1, 1},
//	{ 0, 1, 1},
//	{ 1, 1, 1}
//};
//
//// Two Negative Dimension Expansion
//__constant__ static const char3 voxLookup18[18] =
//{
//	{-1, -1, 0},
//	{ 0, -1, 0},
//	{ 1, -1, 0},
//
//	{-1,  0, 0},
//	{ 0,  0, 0},
//	{ 1,  0, 0},
//		  
//	{-1,  1, 0},
//	{ 0,  1, 0},
//	{ 1,  1, 0},
//
//	{-1, -1, 1},
//	{ 0, -1, 1},
//	{ 1, -1, 1},
//
//	{-1,  0, 1},
//	{ 0,  0, 1},
//	{ 1,  0, 1},
//		 
//	{-1,  1, 1},
//	{ 0,  1, 1},
//	{ 1,  1, 1}
//};
//
//// All Parent Neigbour Expansion (Worst Case)
//__constant__ static const char3 voxLookup27[27] =
//{
//	{-1, -1, -1},
//	{ 0, -1, -1},
//	{ 1, -1, -1},
//
//	{-1,  0, -1},
//	{ 0,  0, -1},
//	{ 1,  0, -1},
//		  
//	{-1,  1, -1},
//	{ 0,  1, -1},
//	{ 1,  1, -1},
//
//	{-1, -1,  0},
//	{ 0, -1,  0},
//	{ 1, -1,  0},
//			  
//	{-1,  0,  0},
//	{ 0,  0,  0},
//	{ 1,  0,  0},
//		 	  
//	{-1,  1,  0},
//	{ 0,  1,  0},
//	{ 1,  1,  0},
//			  
//	{-1, -1,  1},
//	{ 0, -1,  1},
//	{ 1, -1,  1},
//			  
//	{-1,  0,  1},
//	{ 0,  0,  1},
//	{ 1,  0,  1},
//		  	  
//	{-1,  1,  1},
//	{ 0,  1,  1},
//	{ 1,  1,  1}
//};
//
//// VoxLookup Tables
//__constant__ static const int8_t voxLookupSizes[4] = {8, 12, 18, 27};
//__constant__ static const char3* voxLookupTables[4] = {voxLookup8, voxLookup12, voxLookup18, voxLookup27};

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
	CSVONode* node = denseLevel.gLevelNodes + DenseIndex(denseLevelId, octreeParams.DenseSize);

	// Iterate untill level (This portion's nodes should be allocated)
	for(uint32_t i = octreeParams.DenseLevel + 1; i <= level; i++)
	{
		bool allocated;
		uint32_t allocNode = AtomicAllocateNode(allocated, node, gLevelAllocators + i);

		if(allocated && writeId)
		{
			// Write Nodeid to ParentLoc
			uint32_t parentLoc = node - gSVOLevels[i - 1].gLevelNodes;
			int3 parentVoxId = CalculateParentVoxId(voxelId, i - 1, level);
			uint32_t packedParent = PackNodeId(parentVoxId, i - 1,
											   octreeParams.CascadeCount,
											   octreeParams.CascadeBaseLevel,
											   octreeParams.MaxSVOLevel);
			gSVOLevels[i - 1].gVoxId[parentLoc] = packedParent;
		}
		
		assert(allocNode < gLevelCapacities[i]);
		uint32_t childId = CalculateLevelChildId(voxelId, i, level);
		node = gSVOLevels[i].gLevelNodes + allocNode + childId;
	}
	return node - gSVOLevels[level].gLevelNodes;
}