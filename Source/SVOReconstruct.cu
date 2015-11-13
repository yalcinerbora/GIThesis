#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CVoxel.cuh"

inline __device__ CSVOColor AtomicColorNormalAvg(CSVOMaterial* gMaterial, 
												 CSVOColor color,
												 CVoxelNorm voxelNormal)
{
	float4 colorUnpack = UnpackSVOColor(color);
	float3 normalUnpack = ExpandOnlyNormal(voxelNormal);

	CSVOMaterial assumed, old = *gMaterial;
	do
	{
		assumed = old;
		
		// CAS Average
		CSVOColor avgColorPacked;
		CSVOColor avgNormalPacked;
		UnpackSVOMaterial(avgColorPacked, avgNormalPacked, assumed);
		float4 avgColor = UnpackSVOColor(avgColorPacked);
		float3 avgNormal = ExpandOnlyNormal(avgNormalPacked);

		// Averaging (color.w is number of nodes)
		assert(avgColor.w < 255.0f);

		float ratio = avgColor.w / (avgColor.w + 1.0f);

		// New Color Average
		avgColor.x = (ratio * avgColor.x) + (colorUnpack.x / (avgColor.w + 1.0f));
		avgColor.y = (ratio * avgColor.y) + (colorUnpack.y / (avgColor.w + 1.0f));
		avgColor.z = (ratio * avgColor.z) + (colorUnpack.z / (avgColor.w + 1.0f));

		// New Normal Average
		avgNormal.x = (ratio * avgNormal.x) + (normalUnpack.x / (avgColor.w + 1.0f));
		avgNormal.y = (ratio * avgNormal.y) + (normalUnpack.y / (avgColor.w + 1.0f));
		avgNormal.z = (ratio * avgNormal.z) + (normalUnpack.z / (avgColor.w + 1.0f));
		
		avgColorPacked = PackSVOColor(avgColor);
		avgNormalPacked = PackOnlyVoxNorm(avgNormal);
		
		old = atomicCAS(gMaterial, assumed, PackSVOMaterial(avgColorPacked, avgNormalPacked));
	}
	while(assumed != old);
	return old;
}

inline __device__ unsigned int AtomicAllocateNode(CSVONode* gNode,
												  unsigned int& gSVOAllocLocation,
												  unsigned int* gLevelNodeCounts,
												  const unsigned int levelIndex,
												  const unsigned int svoTotalSize)
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

	// much cooler version can be warp level exchange intrinsics
	// which slightly reduces atomic pressure on the single node (on lower tree levels atleast)

	CSVONode old = 0xFFFFFFFE;
	while(old == 0xFFFFFFFE)
	{
		old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE);
		if(old == 0xFFFFFFFF)
		{
			// Allocate
			unsigned int location = atomicAdd(&gSVOAllocLocation, 8); assert(location < svoTotalSize);
			atomicAdd(&gLevelNodeCounts[levelIndex], 8);
			atomicExch(gNode, location);
			old = location;
		}
		__threadfence();
	}
	return old;
}

__global__ void SVOReconstructDetermineNode(CSVONode* gSVODense,
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
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
									  svoConstants.numCascades);
	uint3 denseIndex = CalculateLevelVoxId(voxelPos, svoConstants.denseDepth,
										   svoConstants.totalDepth);

	assert(denseIndex.x < svoConstants.denseDim &&
		   denseIndex.y < svoConstants.denseDim &&
		   denseIndex.z < svoConstants.denseDim);

	// Signal alloc
	*(gSVODense +
	  svoConstants.denseDim * svoConstants.denseDim * denseIndex.z +
	  svoConstants.denseDim * denseIndex.y +
	  denseIndex.x) = 1;
}

__global__ void SVOReconstructDetermineNode(CSVONode* gSVOSparse,
											cudaTextureObject_t tSVODense,
											const CVoxelPage* gVoxelData,

											// Constants
											const unsigned int cascadeNo,
											const unsigned int levelDepth,
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
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
									  svoConstants.numCascades);

	
	assert(currentNode != 0xFFFFFFFF);
	unsigned int nodeIndex = 0;
	for(unsigned int i = svoConstants.denseDepth; i < levelDepth; i++)
	{		
		CSVONode currentNode;
		if(i == svoConstants.denseDepth)
		{
			uint3 denseIndex = CalculateLevelVoxId(voxelPos, svoConstants.denseDepth,
												   svoConstants.totalDepth);

			assert(denseIndex.x < svoConstants.denseDim &&
				   denseIndex.y < svoConstants.denseDim &&
				   denseIndex.z < svoConstants.denseDim);

			currentNode = tex3D<unsigned int>(tSVODense,
											  denseIndex.x,
											  denseIndex.y,
											  denseIndex.z);
		}
		else
		{
			currentNode = gSVOSparse[nodeIndex];
		}

		// Offset according to children
		assert(currentNode != 0xFFFFFFFF);
		unsigned int childIndex = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
		nodeIndex = currentNode + childIndex;
	}

	// Finally Write
	gSVOSparse[nodeIndex] = 1;
}

__global__ void SVOReconstructAllocateLevel(CSVONode* gSVO,
											unsigned int* gLevelNodeCounts,
											unsigned int& gSVOAllocLocation,

											unsigned int svoLevelOffset,
											const unsigned int svoTotalSize,
											const unsigned int level,
											const unsigned int levelSize,
											const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= levelSize) return;

	CSVONode node = gSVO[globalId + svoLevelOffset];
	if(node != 1) return;

	// Allocation
	unsigned int location = atomicAdd(&gSVOAllocLocation, 8); assert(location < svoTotalSize);
	atomicAdd(&gLevelNodeCounts[level - svoConstants.denseDepth + 1], 8);
	gSVO[globalId + svoLevelOffset] = location;
}

__global__ void SVOReconstructMaterialLeaf(CSVOMaterial* gSVOMat,

										   // Const SVO Data
										   const CSVONode* gSVOSparse,
										   cudaTextureObject_t tSVODense,
										   const CVoxelPage* gVoxelData,
										   const unsigned int* gLevelLookupTable,

										   // For Color Lookup
										   CVoxelRender** gVoxelRenderData,

										   // Constants
										   const unsigned int matSparseOffset,
										   const unsigned int cascadeNo,
										   const unsigned int levelDepth,
										   const bool average,
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
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked,
									  cascadeNo,
									  svoConstants.numCascades);
	unsigned char childBit = CalculateLevelChildBit(voxelPos,
													levelDepth + 1,
													svoConstants.totalDepth);
	uint3 levelVoxelId = CalculateLevelVoxId(voxelPos,
											 levelDepth,
											 svoConstants.totalDepth);

	uint3 denseIndex = CalculateLevelVoxId(levelVoxelId, svoConstants.denseDepth,
										   levelDepth);

	assert(denseIndex.x < svoConstants.denseDim &&
		   denseIndex.y < svoConstants.denseDim &&
		   denseIndex.z < svoConstants.denseDim);

	CSVONode currentNode = tex3D<unsigned int>(tSVODense,
											   denseIndex.x,
											   denseIndex.y,
											   denseIndex.z);
	assert(currentNode != 0);
	unsigned int nodeIndex = 0;
	for(unsigned int i = svoConstants.denseDepth + 1; i <= levelDepth; i++)
	{
		unsigned int levelBase = gLevelLookupTable[i - svoConstants.denseDepth - 1];

		unsigned char childBits;
		unsigned int childrenStart;
		UnpackNode(childrenStart, childBits, currentNode);
		assert(childBits != 0);

		// Jump to Next Node
		unsigned char requestedChild = CalculateLevelChildBit(levelVoxelId, i, levelDepth);
		unsigned char childIndex = CalculateChildIndex(childBits, requestedChild);

		nodeIndex = levelBase + childrenStart + childIndex;

		// Last gMem read unnecessary
		if(i < levelDepth) currentNode = gSVOSparse[nodeIndex];
	}

	// Finally found location
	// Average color and normal
	// Fetch obj Id to get color
	ushort2 objectId;
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][voxelId].color);

	if(average)
	{
		// Actual Atomic Average
		AtomicColorNormalAvg(gSVOMat + nodeIndex + matSparseOffset,
							 voxelColorPacked, 
							 voxelNormPacked);
	}
	else
	{
		gSVOMat[nodeIndex + matSparseOffset] = PackSVOMaterial(voxelColorPacked, voxelNormPacked);
	}
}

__global__ void SVOReconstructAverageNode(CSVOMaterial* parentMats,
										  const CSVOMaterial* childrenMats,
										  const CSVONode* gSVONode,
										  const unsigned int parentLevel,
										  const CSVOConstants& svoConstants)
{

}

__global__ void SVOReconstruct(CSVOMaterial* gSVOMat,
							   CSVONode* gSVOSparse,
							   CSVONode* gSVODense,
							   unsigned int* gLevelNodeCounts,
							   unsigned int& gSVOAllocLocation,

							   // For Color Lookup
							   const CVoxelPage* gVoxelData,
							   CVoxelRender** gVoxelRenderData,

							   const unsigned int matSparseOffset,
							   const unsigned int svoTotalSize,
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
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked,
									  cascadeNo,
									  svoConstants.numCascades);

	unsigned int location;
	unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
	for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
	{
		unsigned int childId = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
		
		CSVONode* node = nullptr;
		if(i == svoConstants.denseDepth)
		{
			uint3 levelVoxId = CalculateLevelVoxId(voxelPos, i, svoConstants.totalDepth);
			node = gSVODense + svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
								svoConstants.denseDim * levelVoxId.y +
								levelVoxId.x;
		}
		else
		{
			node = gSVOSparse + location;
		}

		// Allocate (or acquire) next location
		unsigned int levelIndex = i + 1 - svoConstants.denseDepth;
		location = AtomicAllocateNode(node, gSVOAllocLocation, gLevelNodeCounts,
									  levelIndex, svoTotalSize);
		location += childId;
	}

	// We are at bottom of the location can write colors (---)
	ushort2 objectId;
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][voxelId].color);
	AtomicColorNormalAvg(gSVOMat + matSparseOffset + location,
						 voxelColorPacked,
						 voxelNormPacked);
}