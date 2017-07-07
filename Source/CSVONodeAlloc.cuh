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

__device__ unsigned int AtomicAllocateNode(CSVONode* gNode,
										   unsigned int& gLevelAllocator,
										   const CVoxelPos& voxPos);
__device__ const CSVONode* TraverseNode(uint32_t& traversedLevel,
										// SVO
										const CSVOLevelConst* svoLevels,
										// Node Related
										const uint3& voxelId,
										// Constants
										const OctreeParameters& octreeParams,
										const uint32_t level);
__device__ CSVOIllumination* TraverseAndAllocate(// SVO
												 uint32_t* gLevelAllocators,
												 const uint32_t* gLevelCapacities,
												 const CSVOLevel* gSVOLevels,
												 // Node Related
												 const uint3& voxelId,
												 // Constants
												 const OctreeParameters& octreeParams,
												 const uint32_t level);

inline __device__ unsigned int AtomicAllocateNode(CSVONode* gNode,
												  unsigned int& gLevelAllocator,
												  const CVoxelPos& voxPos)
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
	if(gNode->next < 0xFFFFFFFE) return gNode->next;

	// Try to lock the node and allocate for that node
	unsigned int old = 0xFFFFFFFE;
	while(old == 0xFFFFFFFE)
	{
		old = atomicCAS(&gNode->next, 0xFFFFFFFF, 0xFFFFFFFE);
		if(old == 0xFFFFFFFF)
		{
			// Allocate
			unsigned int location = atomicAdd(&gLevelAllocator, 8);
			reinterpret_cast<volatile uint32_t&>(gNode->next) = location;
			gNode->pos = voxPos;
			old = location;
		}
		__threadfence();	// This is important somehow compiler changes this and makes infinite loop on same warp threads
	}
	return old;
}

inline __device__ const CSVONode* TraverseNode(uint32_t& traversedLevel,
											   // SVO
											   const CSVOLevelConst* svoLevels,
											   // Node Related
											   const uint3& voxelId,
											   // Constants
											   const OctreeParameters& octreeParams,
											   const uint32_t level)
{
	// Returns Node Location on That Level	
	uint3 denseLevelId = CalculateParentVoxId(voxelId, octreeParams.DenseLevel, level);
	const CSVOLevelConst& denseLevel = svoLevels[octreeParams.DenseLevel];
	const CSVONode* node = denseLevel.gLevelNodes + DenseIndex(denseLevelId, octreeParams.DenseSize);

	// Iterate untill level (This portion's nodes should be allocated)
	uint32_t i;
	for(i = octreeParams.DenseLevel + 1; i <= level; i++)
	{
		const unsigned int nextNode = node->next;
		if(nextNode == 0xFFFFFFFF) break;

		unsigned int childId = CalculateLevelChildId(voxelId, i, level);
		node = svoLevels[i].gLevelNodes + nextNode + childId;
	}

	traversedLevel = i;
	return node;
}

inline __device__ CSVOIllumination* TraverseAndAllocate(// SVO
														uint32_t* gLevelAllocators,
														const uint32_t* gLevelCapacities,
														const CSVOLevel* gSVOLevels,
														// Node Related
														const uint3& voxelId,
														// Constants
														const OctreeParameters& octreeParams,
														const uint32_t level)
{
	// Returns Node Location on That Level	
	uint3 denseLevelId = CalculateParentVoxId(voxelId, octreeParams.DenseLevel, level);
	const CSVOLevel& denseLevel = gSVOLevels[octreeParams.DenseLevel];
	CSVONode* node = denseLevel.gLevelNodes + DenseIndex(denseLevelId, octreeParams.DenseSize);

	// Iterate untill level (This portion's nodes should be allocated)
	for(uint32_t i = octreeParams.DenseLevel + 1; i <= level; i++)
	{		
		CVoxelPos levelVoxelId = PackNodeId(voxelId, i,
											octreeParams.CascadeCount,
											octreeParams.CascadeBaseLevel,
											octreeParams.MaxSVOLevel);
		unsigned int allocNode = AtomicAllocateNode(node, gLevelAllocators[i], levelVoxelId);
		assert(allocNode < gLevelCapacities[i]);

		unsigned int childId = CalculateLevelChildId(voxelId, i, level);
		node = gSVOLevels[i].gLevelNodes + allocNode + childId;
	}

	ptrdiff_t diff = node - gSVOLevels[level].gLevelNodes;
	return gSVOLevels[level].gLevelIllum + diff;
}