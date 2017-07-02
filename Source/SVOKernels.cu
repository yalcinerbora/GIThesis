#include "SVOKernels.cuh"
#include "GISparseVoxelOctree.h"
#include "GIVoxelPages.h"
#include "CSVOHash.cuh"
#include "CVoxelFunctions.cuh"
#include "CSVOLightInject.cuh"
#include "CSVOIllumAverage.cuh"
#include <cuda.h>

// No Negative Dimension Expansion (Best case)
__constant__ static const char3 voxLookup8[8] =
{
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 0},

	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1}
};

// Single Negative Dimension Expansion
__constant__ static const char3 voxLookup12[12] =
{
	{-1, 0, 0},
	{ 0, 0, 0},
	{ 1, 0, 0},

	{-1, 1, 0},
	{ 0, 1, 0},
	{ 1, 1, 0},

	{-1, 0, 1},
	{ 0, 0, 1},
	{ 1, 0, 1},

	{-1, 1, 1},
	{ 0, 1, 1},
	{ 1, 1, 1}
};

// Two Negative Dimension Expansion
__constant__ static const char3 voxLookup18[18] =
{
	{-1, -1, 0},
	{ 0, -1, 0},
	{ 1, -1, 0},

	{-1,  0, 0},
	{ 0,  0, 0},
	{ 1,  0, 0},
		  
	{-1,  1, 0},
	{ 0,  1, 0},
	{ 1,  1, 0},

	{-1, -1, 1},
	{ 0, -1, 1},
	{ 1, -1, 1},

	{-1,  0, 1},
	{ 0,  0, 1},
	{ 1,  0, 1},
		 
	{-1,  1, 1},
	{ 0,  1, 1},
	{ 1,  1, 1}
};

// All Parent Neigbour Expansion (Worst Case)
__constant__ static const char3 voxLookup27[27] =
{
	{-1, -1, -1},
	{ 0, -1, -1},
	{ 1, -1, -1},

	{-1,  0, -1},
	{ 0,  0, -1},
	{ 1,  0, -1},
		  
	{-1,  1, -1},
	{ 0,  1, -1},
	{ 1,  1, -1},

	{-1, -1,  0},
	{ 0, -1,  0},
	{ 1, -1,  0},
			  
	{-1,  0,  0},
	{ 0,  0,  0},
	{ 1,  0,  0},
		 	  
	{-1,  1,  0},
	{ 0,  1,  0},
	{ 1,  1,  0},
			  
	{-1, -1,  1},
	{ 0, -1,  1},
	{ 1, -1,  1},
			  
	{-1,  0,  1},
	{ 0,  0,  1},
	{ 1,  0,  1},
		  	  
	{-1,  1,  1},
	{ 0,  1,  1},
	{ 1,  1,  1}
};

// VoxLookup Tables
__constant__ static const int8_t voxLookupSizes[4] = {8, 12, 18, 27};
__constant__ static const char3* voxLookupTables[4] = {voxLookup8, voxLookup12, voxLookup18, voxLookup27};

inline __device__ unsigned int AtomicAllocateNode(CSVONode* gNode, unsigned int& gLevelAllocator)
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
    if(*gNode < 0xFFFFFFFE) return *gNode;

	// Try to lock the node and allocate for that node
    CSVONode old = 0xFFFFFFFE;
    while(old == 0xFFFFFFFE)
    {
        old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE);
        if(old == 0xFFFFFFFF)
        {
            // Allocate
            unsigned int location = atomicAdd(&gLevelAllocator, 8);
            *reinterpret_cast<volatile CSVONode*>(gNode) = location;
            old = location;
        }
        __threadfence();	// This is important somehow compiler changes this and makes infinite loop on same warp threads
    }
    return old;
}

inline __device__ const CSVONode* TraverseNode(// SVO
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
	for(uint32_t i = octreeParams.DenseLevel + 1; i <= level; i++)
	{
		unsigned int childId = CalculateLevelChildId(voxelId, i, level);
		node = svoLevels[i].gLevelNodes + *node + childId;
	}
	return node;
}

inline __device__ unsigned int TraverseAndAllocate(// SVO
												   uint32_t& gLevelAllocator,
												   const uint32_t gLevelCapacity,
												   const CSVOLevel* svoLevels,
												   // Node Related
												   const uint3& parentVoxelId,
												   // Constants
												   const OctreeParameters& octreeParams,
												   const uint32_t parentLevel)
{

	const CSVONode* node = TraverseNode(reinterpret_cast<const CSVOLevelConst*>(svoLevels),
										parentVoxelId,
										octreeParams,
										parentLevel);

	// Now Node pointer points the required location
	CSVONode nodeLocation = AtomicAllocateNode(const_cast<CSVONode*>(node), 
											   gLevelAllocator);
	
	// Check the capacity if capacity Fails
	assert(nodeLocation < gLevelCapacity);
	
	// Return Node
	return nodeLocation;
}
//
//
//
//	unsigned int location;
//	CSVONode* node = nullptr;
//	for(unsigned int i = octreeParams.DenseLevel; i <= level; i++)
//	{
//		CSVONode* node = nullptr;
//		if(i == svoConstants.denseDepth)
//		{
//			uint3 levelVoxId = CalculateLevelVoxId(voxelPos, i, svoConstants.totalDepth);
//			node = gSVODense +
//				svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
//				svoConstants.denseDim * levelVoxId.y +
//				levelVoxId.x;
//		}
//		else
//		{
//			node = gSVOSparse + gLevelOffsets[levelIndex] + location;
//		}
//
//		// Allocate (or acquire) next location
//		location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1]);
//		assert(location < gLevelTotalSizes[levelIndex + 1]);
//
//		// Offset child
//		unsigned int childId = CalculateLevelChildId(currentVoxPos, i + 1, svoConstants.totalDepth);
//		location += childId;
//	}
//}
//
////inline __device__ unsigned int FindDenseChildren(const uint3& parentIndex,
////                                                 const unsigned int childId,
////                                                 const unsigned int levelDim)
////{
////    // Go down 1 lvl
////    uint3 childIndex = parentIndex;
////    childIndex.x *= 2;
////    childIndex.y *= 2;
////    childIndex.z *= 2;
////
////    uint3 offsetIndex =
////    {
////        childId % 2,
////        childId / 2,
////        childId / 4
////    };
////    childIndex.x += offsetIndex.x;
////    childIndex.y += offsetIndex.y;
////    childIndex.z += offsetIndex.z;
////
////    unsigned int childLvlDim = levelDim << 1;
////    unsigned int linearChildId = childIndex.z * childLvlDim * childLvlDim +
////        childIndex.y * childLvlDim +
////        childIndex.z;
////    return linearChildId;
////}
////
//
//__global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,
//										  cudaSurfaceObject_t sDenseMat,
//
//										  const CSVONode* gSVODense,
//										  const CSVONode* gSVOSparse,
//
//										  const unsigned int* gLevelOffsets,
//										  const unsigned int& gSVOLevelOffset,
//										  const unsigned int& gSVONextLevelOffset,
//
//										  const unsigned int levelNodeCount,
//										  const unsigned int matOffset,
//										  const unsigned int currentLevel,
//										  const CSVOConstants& svoConstants)
//{
//	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int nodeId = globalId / 2;
//
//	// Cull if out of range
//	if(nodeId > levelNodeCount) return;
//
//	// Read Sibling Materials
//	const CSVONode* n = (currentLevel == svoConstants.denseDepth) ? gSVODense : gSVOSparse;
//	CSVONode node = n[gSVOLevelOffset + nodeId];
//
//	// Cull if there is no node no need to average
//	if(node == 0xFFFFFFFF) return;
//
//	// Only fetch parent when there a potential to have one
//	bool fetchParentMat = ((svoConstants.totalDepth - currentLevel) < svoConstants.numCascades);
//
//	uint64_t parentMat;
//	if(globalId % 2 == 0) parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + nodeId].colorPortion : 0x0;
//	else parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + nodeId].normalPortion : 0x0;
//
//	// Average Portion
//	// Material Data
//	unsigned int count = 0;
//	float4 avgSegment1 = {0.0f, 0.0f, 0.0f, 0.0f};
//	float4 avgSegment2 = {0.0f, 0.0f, 0.0f, 0.0f};
//
//	// Parent Incorporate
//	if(parentMat != 0x0)
//	{
//		if(globalId % 2 == 0)
//		{
//			CSVOColor colorPacked = UnpackSVOMaterialColorOrNormal(parentMat);
//			float4 color = UnpackSVOColor(colorPacked);
//		
//			avgSegment1.x = 8 * color.x;
//			avgSegment1.y = 8 * color.y;
//			avgSegment1.z = 8 * color.z;
//			avgSegment1.w = 8 * color.w;
//		}
//		else
//		{
//			CVoxelNorm normalPacked = UnpackSVOMaterialColorOrNormal(parentMat);
//			float4 normal = UnpackSVONormal(normalPacked);
//
//			avgSegment2.x = 8 * normal.x;
//			avgSegment2.y = 8 * normal.y;
//			avgSegment2.z = 8 * normal.z;
//			avgSegment2.w = 8 * normal.w;
//		}
//		count += 8;
//	}
//
//	// Average
//	if(node != 0xFFFFFFFF)
//	{
//		#pragma unroll
//		for(unsigned int i = 0; i < 8; i++)
//		{
//			unsigned int currentNodeId = node + i;
//			if(globalId % 2 == 0)
//			{
//				uint64_t mat = gSVOMat[matOffset + gSVONextLevelOffset + currentNodeId].colorPortion;
//				if(mat == 0x0) continue;
//
//				CSVOColor colorPacked = UnpackSVOMaterialColorOrNormal(mat);
//				float4 color = UnpackSVOColor(colorPacked);
//
//				avgSegment1.x += color.x;
//				avgSegment1.y += color.y;
//				avgSegment1.z += color.z;
//				avgSegment1.w += color.w;
//			}
//			else
//			{
//				uint64_t mat = gSVOMat[matOffset + gSVONextLevelOffset + currentNodeId].normalPortion;
//				if(mat == 0x0) continue;
//
//				CVoxelNorm normalPacked = UnpackSVOMaterialColorOrNormal(mat);
//				float4 normal = UnpackSVONormal(normalPacked);
//
//				avgSegment2.x += normal.x;
//				avgSegment2.y += normal.y;
//				avgSegment2.z += normal.z;
//				avgSegment2.w += normal.w;
//			}
//			count++;
//		}
//	}
//
//	// Divide by Count
//	if(count == 0) count = 1.0f;
//	float countInv = 1.0f / static_cast<float>(count);
//	avgSegment1.x *= countInv;
//	avgSegment1.y *= countInv;
//	avgSegment1.z *= countInv;
//	avgSegment1.w *= countInv;
//
//	avgSegment2.x *= countInv;
//	avgSegment2.y *= countInv;
//	avgSegment2.z *= countInv;
//	avgSegment2.w *= (count > 8) ? 0.0625f : 0.125f;
//
//	// Pack and Store	
//	uint64_t averageValue;
//	if(globalId % 2 == 0)
//	{
//		CSVOColor colorPacked = PackSVOColor(avgSegment1);		
//		averageValue = PackSVOMaterialPortion(colorPacked, 0x0);
//	}
//	else
//	{		
//		CVoxelNorm normPacked = PackSVONormal(avgSegment2);
//		averageValue = PackSVOMaterialPortion(normPacked, 0x0);
//	}
//	
//    if(currentLevel == svoConstants.denseDepth)
//    {
//        int3 dim =
//        {
//            static_cast<int>(nodeId % svoConstants.denseDim),
//            static_cast<int>((nodeId / svoConstants.denseDim) % svoConstants.denseDim),
//            static_cast<int>(nodeId / (svoConstants.denseDim * svoConstants.denseDim))
//        };
//        uint2 data =
//        {
//            static_cast<unsigned int>(averageValue & 0x00000000FFFFFFFF),
//            static_cast<unsigned int>(averageValue >> 32)
//        };
//		int dimX = (globalId % 2 == 0) ? (dim.x * sizeof(uint4)) : (dim.x * sizeof(uint4) + sizeof(uint2));
//        surf3Dwrite(data, sDenseMat, dimX, dim.y, dim.z);
//    }
//    else
//    {
//		if(globalId % 2 == 0) gSVOMat[matOffset + gSVOLevelOffset + nodeId].colorPortion = averageValue;
//		else gSVOMat[matOffset + gSVOLevelOffset + nodeId].normalPortion = averageValue;
//    }
//}
//
//__global__ void SVOReconstructAverageNode(cudaSurfaceObject_t sDenseMatChild,
//                                          cudaSurfaceObject_t sDenseMatParent,
//
//                                          const unsigned int parentSize)
//{
//    // Linear Id
//    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned int parentId = globalId / GI_DENSE_WORKER_PER_PARENT;
//
//    // 3D Id
//    char3 idMap = voxLookup[globalId % GI_DENSE_WORKER_PER_PARENT];
//    uint3 parentId3D =
//    {
//        static_cast<unsigned int>(parentId % parentSize),
//        static_cast<unsigned int>((parentId / parentSize) % parentSize),
//        static_cast<unsigned int>(parentId / (parentSize * parentSize))
//    };
//    uint3 childId3D =
//    {
//        parentId3D.x * 2 + idMap.x,
//        parentId3D.y * 2 + idMap.y,
//        parentId3D.z * 2 + idMap.z
//    };
//
//    // 3D Fetch
//    uint4 data;
//    surf3Dread(&data, sDenseMatChild,
//               childId3D.x * sizeof(uint4),
//               childId3D.y,
//               childId3D.z);
//
//    // Data
//    unsigned int count = (data.x == 0 && 
//						  data.y == 0 && 
//						  data.z == 0 && 
//						  data.w == 0) ? 0 : 1;
//    float4 color = UnpackSVOColor(data.x);
//    float4 normal = UnpackSVONormal(data.w);
//
//    // Average	
//    #pragma unroll
//    for(int offset = GI_DENSE_WORKER_PER_PARENT / 2; offset > 0; offset /= 2)
//    {
//        color.x += __shfl_down(color.x, offset, GI_DENSE_WORKER_PER_PARENT);
//        color.y += __shfl_down(color.y, offset, GI_DENSE_WORKER_PER_PARENT);
//        color.z += __shfl_down(color.z, offset, GI_DENSE_WORKER_PER_PARENT);
//        color.w += __shfl_down(color.w, offset, GI_DENSE_WORKER_PER_PARENT);
//
//        normal.x += __shfl_down(normal.x, offset, GI_DENSE_WORKER_PER_PARENT);
//        normal.y += __shfl_down(normal.y, offset, GI_DENSE_WORKER_PER_PARENT);
//        normal.z += __shfl_down(normal.z, offset, GI_DENSE_WORKER_PER_PARENT);
//        normal.w += __shfl_down(normal.w, offset, GI_DENSE_WORKER_PER_PARENT);
//
//        count += __shfl_down(count, offset, GI_DENSE_WORKER_PER_PARENT);
//    }
//
//    // Division
//    float countInv = 1.0f / ((count != 0) ? float(count) : 1.0f);
//    color.x *= countInv;
//    color.y *= countInv;
//    color.z *= countInv;
//    color.w *= countInv;
//
//    normal.x *= countInv;
//    normal.y *= countInv;
//    normal.z *= countInv;
//	normal.w *= 0.125f;
//
//    data.x = PackSVOColor(color);
//    data.w = PackSVONormal(normal);
//
//    if(globalId % GI_DENSE_WORKER_PER_PARENT == 0 && count != 0)
//    {
//        surf3Dwrite(data, sDenseMatParent,
//                    parentId3D.x * sizeof(uint4),
//                    parentId3D.y,
//                    parentId3D.z);
//    }
//}
//
__global__ void SVOReconstruct(// SVO
							   CSVOLevel* gSVOLevels,
							   uint32_t* gLevelAllocators,
							   const uint32_t* gLevelCapacities,
							   // Voxel Pages
							   const CVoxelPageConst* gVoxelPages,
							   const CVoxelGrid* gGridInfos,
							   // Cache Data (for Voxel Albedo)
							   const BatchVoxelCache* gBatchVoxelCache,
							   // Light Injection Related
							   const CLightInjectParameters liParams,
							   // Limits
							   const OctreeParameters octreeParams,
							   const uint32_t batchCount)
{

	// Shared Memory for generic data
	__shared__ CSegmentInfo sSegInfo;
	__shared__ CMeshVoxelInfo sMeshVoxelInfo;

	// Meta Nodes
	// and their expansion policy (LSB 3 bits 1 means do not expand in negative direction)
	constexpr int32_t HashSize = /*263;*//*523;*//*1031;*/2039;
	__shared__ uint32_t sHashSpotAllocator;
	__shared__ CVoxelPos sMetaNodes[HashSize];
	__shared__ uint32_t sMetaNodeBitmap[HashSize];
	__shared__ uint32_t sOccupiedHashSpots[HashSize];

	unsigned int blockLocalId = threadIdx.x;
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GIVoxelPages::PageSize;
	unsigned int pageLocalId = globalId % GIVoxelPages::PageSize;
	unsigned int pageLocalSegmentId = pageLocalId / GIVoxelPages::SegmentSize;
	unsigned int segmentLocalVoxId = pageLocalId % GIVoxelPages::SegmentSize;

	// Get Segments Obj Information Struct
	CObjectType objType;
	CSegmentOccupation occupation;
	uint8_t cascadeId;
	bool firstOccurance;
	if(blockLocalId == 0)
	{
		// Load to smem
		// Todo split this into the threadss
		sSegInfo = gVoxelPages[pageId].dSegmentInfo[pageLocalSegmentId];
		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	}
	__syncthreads();
	if(blockLocalId != 0)
	{
		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	}
	// Full Block Cull
	if(occupation == CSegmentOccupation::EMPTY) return;
	assert(occupation != CSegmentOccupation::MARKED_FOR_CLEAR);
	if(blockLocalId == 0)
	{
		sMeshVoxelInfo = gBatchVoxelCache[cascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId];
	}
	__syncthreads();

	// Fetch Position and Normal
	const CVoxelPos voxelPosPacked = gVoxelPages[pageId].dGridVoxPos[pageLocalId];
	const CVoxelNorm voxelNormPacked = gVoxelPages[pageId].dGridVoxNorm[pageLocalId];
	
	// Unpack Position
	uint3 voxPos = ExpandVoxPos(voxelPosPacked);
	uint3 nodePos = ExpandToSVODepth(voxPos,
									 cascadeId,
									 octreeParams.CascadeCount,
									 octreeParams.CascadeBaseLevel);
	
	// From now on there is extremely heavy work per thread (or multi-thread)
	// Each thread will create 8 neigburing sample voxels
	// All of those voxels may require additional nodes (up to 8)
	// Which means worst case (64 nodes (27 parents) will be required to be generated per pageVoxel)

	// Some sort of filtering/reduction is mandatory performance-wise and reduction
	// should generate improvements since many page voxels are adjacent to each other
	// (they represent same object thus should spatially be closer))
	
	//if(blockLocalId == 0) printf("Object Voxel Count %d\n", sMeshVoxelInfo.voxCount);

	// Construct Level By Level
	uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - cascadeId;
	for(uint32_t i = octreeParams.DenseLevel + 1; i <= cascadeMaxLevel; i++)
	{
		//if(blockLocalId == 0) printf("Level %d -----\n", i);

		// Before Hash Resolve Initialize Hash Tables
		HashTableReset(sHashSpotAllocator,
					   sMetaNodes,
					   sMetaNodeBitmap,
					   HashSize);
		__syncthreads();

		// Hash this Meta node (reduction)
		// Only hash if your node is valid (which is determined by normal)
		if(voxelNormPacked != 0xFFFFFFFF)
		{
			// Determine Meta Node of this level and
			// Determine Bitmap
			uint3 levelNodePos = CalculateParentVoxId(nodePos, i, cascadeMaxLevel);
			uint3 metaNode;
			metaNode.x = levelNodePos.x & 0xFFFFFFFE;
			metaNode.y = levelNodePos.y & 0xFFFFFFFE;
			metaNode.z = levelNodePos.z & 0xFFFFFFFE;
			uint32_t bitmap = 0x00;
			bitmap |= (levelNodePos.z & 0x1) << 2;
			bitmap |= (levelNodePos.y & 0x1) << 1;
			bitmap |= (levelNodePos.x & 0x1) << 0;
			CVoxelPos metaNodePacked = PackNodeId(metaNode, i, 
												  octreeParams.CascadeCount,
												  octreeParams.CascadeBaseLevel,
												  octreeParams.MaxSVOLevel);	

			//if(blockIdx.x == 0)
			//{
			//	uint3 unpack = ExpandVoxPos(voxelPosPacked);
			//	printf("Voxel Pos (%d %d %d) "
			//		   "Node Pos (%d %d %d) "
			//		   "Level Pos (%d %d %d) "
			//		   "Meta Pos (%d %d %d) "
			//		   "After Pack (%#010X) "
			//		   "Bitmap (%#010X)"
			//		   "\n",					   
			//		   unpack.x, unpack.y, unpack.z,
			//		   nodePos.x, nodePos.y, nodePos.z,
			//		   levelNodePos.x, levelNodePos.y, levelNodePos.z,
			//		   metaNode.x, metaNode.y, metaNode.z,
			//		   metaNodePacked,
			//		   bitmap);
			//}
			
			// Hashing
			HashMap(// Hash table
					sMetaNodes,
					sMetaNodeBitmap,
					// Linear storage of occupied locations of hash table
					sOccupiedHashSpots,
					sHashSpotAllocator,
					// Key value
					metaNodePacked,
					bitmap,
					// Hash table limits
					HashSize);
		}
		__syncthreads();
		
		//if(blockLocalId == 0) printf("-----\n");
		//if(blockIdx.x == 0 &&
		//   blockLocalId == 0)
		//{			
		//	printf("Level #%d Meta Node Count %d\n", i, sHashSpotAllocator);
		//}

		// We reduced nodes to some degree
		// Now each 8 thread will be responsible 
		// for allocating a Meta node
		// First entire block will loop around nodes
		constexpr uint32_t threadPerMetaNode = 8;
		const uint32_t workerCount = blockDim.x / threadPerMetaNode;
		const uint32_t iterationCount = (sHashSpotAllocator + workerCount - 1) / workerCount;
		assert(blockDim.x % threadPerMetaNode == 0);
		for(int k = 0; k < iterationCount; k++)
		{
			uint32_t hashId = k * workerCount + blockLocalId / threadPerMetaNode;
			uint32_t hashLocalId = blockLocalId % threadPerMetaNode;
			if(hashId < sHashSpotAllocator)
			{
				// Meta Node
				CVoxelPos reducedMetaNode = sMetaNodes[sOccupiedHashSpots[hashId]];
				uint32_t metaNodeBits = sMetaNodeBitmap[sOccupiedHashSpots[hashId]] & 0x00000007;
				uint32_t invMetaNodeBits = (~metaNodeBits) & 0x00000007;
				
				// Expand Node
				uint3 expandedNode = ExpandToSVODepth(ExpandVoxPos(reducedMetaNode),
													  octreeParams.MaxSVOLevel - i,
													  octreeParams.CascadeCount,
													  octreeParams.CascadeBaseLevel);

				// Find out generation size and lookup tables
				int8_t lookupIndex = static_cast<int8_t>(__popc(invMetaNodeBits));
				const char3* lookupTable = voxLookupTables[lookupIndex];
				const int8_t lookupCount = voxLookupSizes[lookupIndex];

				// Determine Swap Locations
				// For one or two negative expansions lookup table
				// needs to be adjusted since it stored as x (or xy) negative
				// we need to check bits for that
				int8_t swapFrom = 0, swapTo = 0;
				if(__popc(metaNodeBits) == 1)
				{
					swapFrom = 2;
					swapTo = __ffs(metaNodeBits) - 1;
				}
				else if(__popc(metaNodeBits) == 2)
				{
					swapTo = __ffs(invMetaNodeBits) - 1;
				}

				//if(blockIdx.x == 0 && 
				//   hashId == 0 &&
				//   hashLocalId == 0)
				//{
				//	printf("Hash Packed Voxel (%#010X) "
				//		   "My Expanded Node (%d %d %d) "
				//		   "Bitmap (%#010X) (%#010X) "
				//		   "Loop Count %d "
				//		   "Lookup Count %d "
				//		   "%d - %d "
				//		   "\n",
				//		   reducedMetaNode,
				//		   expandedNode.x, expandedNode.y, expandedNode.z,
				//		   metaNodeBits, invMetaNodeBits, lookupIndex + 1, lookupCount,
				//		   swapFrom, swapTo);
				//}
				
				for(int8_t j = 0; j < lookupIndex + 1; j++)
				{
					uint32_t hashNodeId = j * threadPerMetaNode + hashLocalId;
					if(hashNodeId >= lookupCount) continue;
					
					// Swap the neigbour map
					char3 neigbourMap = lookupTable[hashNodeId];
					Swap(neigbourMap, swapFrom, swapTo);

					int3 parentNode;
					parentNode.x = static_cast<int>(expandedNode.x >> 1) + neigbourMap.x;
					parentNode.y = static_cast<int>(expandedNode.y >> 1) + neigbourMap.y;
					parentNode.z = static_cast<int>(expandedNode.z >> 1) + neigbourMap.z;

					// Boundary Check
					int levelSize = (0x1 << i);
					if(parentNode.x < 0 || parentNode.x >= levelSize ||
					   parentNode.y < 0 || parentNode.y >= levelSize ||
					   parentNode.z < 0 || parentNode.z >= levelSize)
						continue;

					uint3 uParentNode;
					uParentNode.x = static_cast<uint32_t>(parentNode.x);
					uParentNode.y = static_cast<uint32_t>(parentNode.y);
					uParentNode.z = static_cast<uint32_t>(parentNode.z);
					
					//if(blockIdx.x == 0 && hashId == 0)
					//{
					//	printf("My Parent Node Pos (%d %d %d) "
					//		   " HashNodeId %d "
					//		   "\n",
					//		   parentNode.x, parentNode.y, parentNode.z,
					//		   hashNodeId);
					//}
					
					//atomicAdd(gLevelAllocators + i, 1);
					uint32_t node = TraverseAndAllocate(// SVO
														gLevelAllocators[i],
														gLevelCapacities[i],
														gSVOLevels,
														// Node Related
														uParentNode,
														// Constants
														octreeParams,
														i - 1);
				}

				// Is this required ? (or volatile cast does the trick?)
				//__threadfence_block();
			}
		}
		// This level's nodes are allocated now to the next level
		__syncthreads();
	}

	//__threadfence();
	// All Levels are Allocated (for this block at least)
	// Now each thread will write on its own to the leaf
	// Cull unnecessary threads
	if(voxelNormPacked == 0xFFFFFFFF) return;

	// Now load albeo etc and average those on leaf levels
	// Find your opengl data and voxel cache
	const uint16_t& batchId = sSegInfo.batchId;
	const BatchVoxelCache& batchCache = gBatchVoxelCache[cascadeId * batchCount + batchId];

	// Voxel Ids
	const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
	const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelInfo.voxOffset;
	
	const CVoxelOccupancy voxOccupPacked = gVoxelPages[pageId].dGridVoxOccupancy[pageLocalId];
	const CVoxelAlbedo voxAlbedoPacked = batchCache.dVoxelAlbedo[batchLocalVoxelId];
	
	// Unpack Occupancy
	float3 weights = ExpandOccupancy(voxOccupPacked);
	float3 normal = ExpandVoxNormal(voxelNormPacked);
	float4 voxAlbedo = UnpackSVOIrradiance(*reinterpret_cast<const CSVOIrradiance*>(&voxAlbedoPacked));
	
	// Light Injection
	float4 irradiance;
	float3 lightDir = {0.0f, 0.0f, 0.0f};
	if(liParams.injectOn)
	{
		// World Space Position Reconstruction
		const float3 edgePos = gGridInfos[cascadeId].position;
		const float span = gGridInfos[cascadeId].span;
		float3 worldPos;
		worldPos.x = edgePos.x + (static_cast<float>(voxPos.x) + weights.x) * span;
		worldPos.x = edgePos.y + (static_cast<float>(voxPos.y) + weights.y) * span;
		worldPos.x = edgePos.z + (static_cast<float>(voxPos.z) + weights.z) * span;

		// Generated Irradiance
		float3 irradianceDiffuse = LightInject(lightDir,
											   // Node Params
											   worldPos,
											   voxAlbedo,
											   normal,
											   // Light Parameters
											   liParams);

		irradiance.x = irradianceDiffuse.x;
		irradiance.y = irradianceDiffuse.x;
		irradiance.z = irradianceDiffuse.x;
	}
	else
	{
		irradiance.x = voxAlbedo.x;
		irradiance.y = voxAlbedo.x;
		irradiance.z = voxAlbedo.x;
	}
	irradiance.w = voxAlbedo.w;

	// Now Leaf Illumination Data Injection
	// Each Node will traverse all potential 27 parents
			
	// Determine Meta Node of this level and
	// Determine Bitmap
	uint3 parentNode;
	parentNode.x = nodePos.x & 0xFFFFFFFE;
	parentNode.y = nodePos.y & 0xFFFFFFFE;
	parentNode.z = nodePos.z & 0xFFFFFFFE;
	uint32_t bitmap = 0x00;
	bitmap |= (nodePos.z & 0x1) << 2;
	bitmap |= (nodePos.y & 0x1) << 1;
	bitmap |= (nodePos.x & 0x1) << 0;	
	uint32_t invBitmap = (~bitmap) & 0x00000007;

	// Find out generation size and lookup tables
	int8_t lookupIndex = static_cast<int8_t>(__popc(invBitmap));
	const char3* lookupTable = voxLookupTables[lookupIndex];
	const int8_t lookupCount = voxLookupSizes[lookupIndex];

	// Determine Swap Locations
	// For one or two negative expansions lookup table
	// needs to be adjusted since it stored as x (or xy) negative
	// we need to check bits for that
	int8_t swapFrom = 0, swapTo = 0;
	if(__popc(bitmap) == 1)
	{
		swapFrom = 2;
		swapTo = __ffs(bitmap) - 1;
	}
	else if(__popc(bitmap) == 2)
	{
		swapTo = __ffs(invBitmap) - 1;
	}

	// We found out iterations now iterate
	for(int8_t j = 0; j < lookupCount; j++)
	{
		uint32_t parentLevel = cascadeMaxLevel - 1;

		// Swap the neigbour map
		char3 neigbourMap = lookupTable[j];
		Swap(neigbourMap, swapFrom, swapTo);

		int3 currentParent;
		currentParent.x = static_cast<int>(nodePos.x >> 1) + neigbourMap.x;
		currentParent.y = static_cast<int>(nodePos.y >> 1) + neigbourMap.y;
		currentParent.z = static_cast<int>(nodePos.z >> 1) + neigbourMap.z;

		// Boundary Check
		int parentLevelSize = (0x1 << parentLevel);
		if(currentParent.x < 0 || currentParent.x >= parentLevelSize ||
		   currentParent.y < 0 || currentParent.y >= parentLevelSize ||
		   currentParent.z < 0 || currentParent.z >= parentLevelSize)
			continue;

		uint3 uParentNode;
		uParentNode.x = static_cast<uint32_t>(currentParent.x);
		uParentNode.y = static_cast<uint32_t>(currentParent.y);
		uParentNode.z = static_cast<uint32_t>(currentParent.z);

		// Traverse to this parent
		const CSVONode* n = TraverseNode(reinterpret_cast<CSVOLevelConst*>(gSVOLevels),
										 uParentNode, octreeParams, parentLevel);
			
		//// Found Parent Now This parent may have multiple children (up to 8)
		//// to be allocated
		//1 + (1 - (neigbourMap.x | (bitmap >> 0))) *
		//1 + (1 - (neigbourMap.y | (bitmap >> 1))) *
		//1 + (1 - (neigbourMap.z | (bitmap >> 2))) *

		//int8_t childCount = 1 * 
		//for(int i = 0; i < )
		

		//AtomicIllumAvg(illumNode, irradiance, normal, lightDir, occupancy);

	}
	
	// Finally All Done




	
}

// Old Code
//float totalOccupancy = 0.0f;
//for(unsigned int i = 0; i < GI_SVO_WORKER_PER_NODE; i++)
//{
//	// Create NeigNode
//	uint3 currentVoxPos = voxelPos;
//	unsigned int cascadeOffset = svoConstants.numCascades - cascadeNo - 1;
//	currentVoxPos.x += voxLookup[i].x * (voxOffset.x << cascadeOffset);
//	currentVoxPos.y += voxLookup[i].y * (voxOffset.y << cascadeOffset);
//	currentVoxPos.z += voxLookup[i].z * (voxOffset.z << cascadeOffset);
//	
//	// Calculte this nodes occupancy
//	float occupancy = 1.0f;
//	float3 volume;
//	volume.x = (voxLookup[i].x == 1) ? weights.x : (1.0f - weights.x);
//	volume.y = (voxLookup[i].y == 1) ? weights.y : (1.0f - weights.y);
//	volume.z = (voxLookup[i].z == 1) ? weights.z : (1.0f - weights.z);
//	occupancy = volume.x * volume.y * volume.z;
//	totalOccupancy += occupancy;
//	//printf("(%d, %d, %d) occupancy %f\n",
//	//	   voxLookup[i].z, voxLookup[i].y, voxLookup[i].x,
//	//	   occupancy);
//	unsigned int location;
//	unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
//	for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
//	{
//		unsigned int levelIndex = i - svoConstants.denseDepth;
//		CSVONode* node = nullptr;
//		if(i == svoConstants.denseDepth)
//		{
//			uint3 levelVoxId = CalculateLevelVoxId(currentVoxPos, i, svoConstants.totalDepth);
//			node = gSVODense +
//				svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
//				svoConstants.denseDim * levelVoxId.y +
//				levelVoxId.x;
//		}
//		else
//		{
//			node = gSVOSparse + gLevelOffsets[levelIndex] + location;
//		}
//		// Allocate (or acquire) next location
//		location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1]);
//		assert(location < gLevelTotalSizes[levelIndex + 1]);
//		// Offset child
//		unsigned int childId = CalculateLevelChildId(currentVoxPos, i + 1, svoConstants.totalDepth);
//		location += childId;
//	}
//	AtomicAvg(gSVOMat + matSparseOffset +
//			  gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] + location,
//			  voxelColorPacked,
//			  voxelNormPacked,
//			  {0.0f, 0.0f, 0.0f, 0.0f},
//			  occupancy);
//	//// Non atmoic overwrite
//	//gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 -
//	//		svoConstants.denseDepth] + location].colorPortion = PackSVOMaterialPortion(voxelColorPacked, 0x0);
//	//gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 -
//	//		svoConstants.denseDepth] + location].normalPortion = PackSVOMaterialPortion(voxelNormPacked, 0x0);
//		
//}
////printf("total occupancy %f\n", totalOccupancy);
