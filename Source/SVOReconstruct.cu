#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CVoxel.cuh"
#include <cuda.h>

inline __device__ CSVOMaterial Average(const CSVOMaterial& material,
										const float4& colorUnpack,
										const float3& normalUnpack)
{
	// Unpack Material
	CSVOColor avgColorPacked;
	CSVOColor avgNormalPacked;
	UnpackSVOMaterial(avgColorPacked, avgNormalPacked, material);
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
	return PackSVOMaterial(avgColorPacked, avgNormalPacked);
}

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
		old = atomicCAS(gMaterial, assumed, Average(assumed, colorUnpack, normalUnpack));
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

inline __device__ unsigned int FindDenseChildren(const uint3& parentIndex,
												 const unsigned int childId,
												 const unsigned int levelDim)
{
	// Go down 1 lvl
	uint3 childIndex = parentIndex;
	childIndex.x *= 2;
	childIndex.y *= 2;
	childIndex.z *= 2;

	uint3 offsetIndex =
	{
		childId % 2,
		childId / 2,
		childId / 4
	};
	childIndex.x += offsetIndex.x;
	childIndex.y += offsetIndex.y;
	childIndex.z += offsetIndex.z;

	unsigned int childLvlDim = levelDim << 1;
	unsigned int linearChildId = childIndex.z * childLvlDim * childLvlDim +
								 childIndex.y * childLvlDim +
								 childIndex.z;
	return linearChildId;
}

__global__ void SVOReconstructDetermineNode(CSVONode* gSVODense,
											const CVoxelPage* gVoxelData,

											const unsigned int cascadeNo,
											const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
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
									  svoConstants.numCascades,
									  svoConstants.totalDepth);
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
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
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
									  svoConstants.numCascades,
									  svoConstants.totalDepth);

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

										   // Page Data
										   const CVoxelPage* gVoxelData,
										  
										   // For Color Lookup
										   CVoxelRender** gVoxelRenderData,

										   // Constants
										   const unsigned int matSparseOffset,
										   const unsigned int cascadeNo,
										   const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
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
									  svoConstants.numCascades,
									  svoConstants.totalDepth);


	unsigned int nodeIndex = 0;
	unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
	for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
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

	// Finally found location
	// Average color and normal
	// Fetch obj Id to get color
	ushort2 objectId;
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][voxelId].color);

	// Atomic Average
	AtomicColorNormalAvg(gSVOMat + nodeIndex + matSparseOffset,
						 voxelColorPacked,
						 voxelNormPacked);

	//gSVOMat[nodeIndex + matSparseOffset] = PackSVOMaterial(voxelColorPacked, 
	//													   voxelNormPacked);

}

__global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,

										  const CSVONode* gSVO,

										  const unsigned int matOffset,
										  const unsigned int svoLevelOffset,
										  const unsigned int currentLevel,

										  const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int linearParentId = threadIdx.x + blockIdx.x * (blockDim.x / 4);
	unsigned int globalParentId = globalId / 4;

	unsigned int localNodeId = threadIdx.x / 4;
	unsigned int localNodeChildId = threadIdx.x % 4;

	__shared__ CSVONode sNodeIds[GI_THREAD_PER_BLOCK / 4];
	__shared__ CSVOMaterial sMatAvg[GI_THREAD_PER_BLOCK / 4];

	// Read Sibling Materials
	CSVOMaterial mat[2];
	if(currentLevel < svoConstants.denseDepth)
	{
		// Reduction between dense
		// Find Levels
		unsigned int linearId = globalParentId;
		unsigned int levelDim = svoConstants.denseDim >> (svoConstants.denseDepth - currentLevel);
		uint3 index = 
		{
			linearId % levelDim,
			linearId / levelDim,
			linearId / (levelDim * levelDim)
		};

		// Find Children Ids
		unsigned int childIds[2];
		childIds[0] = FindDenseChildren(index, globalId % 4, levelDim);
		childIds[1] = FindDenseChildren(index, (globalId % 4) + 4, levelDim);

		// Calculate Dense Start Location
		unsigned int lvlOffset = static_cast<unsigned int>((1.0 - powf(8.0f, currentLevel + 1)) / (1.0f - 8.0f));

		// Load Material
		mat[0] = gSVOMat[lvlOffset + childIds[0]];
		mat[1] = gSVOMat[lvlOffset + childIds[1]];
	}
	else if(currentLevel >= svoConstants.denseDepth)
	{
		// Read is from sparse
		// Write is to dense or sparse
		
		// Dense read coalesced
		if(threadIdx.x < blockDim.x / 4)
		{
			sNodeIds[threadIdx.x] = gSVO[svoLevelOffset + linearParentId];
		}
		__syncthreads();

		// Load Children
		CSVONode node[2];
		unsigned int nodeId = sNodeIds[localNodeId] + localNodeChildId;
		node[0] = (sNodeIds[localNodeId] != 0xFFFFFFFF) ? gSVO[nodeId] : 0xFFFFFFFF;
		node[1] = (sNodeIds[localNodeId] != 0xFFFFFFFF) ? gSVO[nodeId + 4] : 0xFFFFFFFF;
		// Each Thread has two children
		// T1 -> 0, 4
		// T2 -> 1, 5
		// T3 -> 2, 6
		// T4 -> 3, 7

		// Load Material
		mat[0] = (node[0] != 0xFFFFFFFF) ? gSVOMat[matOffset + node[0]] : 0;
		mat[1] = (node[1] != 0xFFFFFFFF) ? gSVOMat[matOffset + node[1]] : 0;
	}

	// Average Portion
	// Material Data
	unsigned int count = 0;
	float4 colorAvg = {0, 0, 0, 0};
	float3 normalAvg = {0.0f, 0.0f, 0.0f};
	
	// Average Yours
	for(unsigned int i = 0; i < 2; i++)
	{
		if(mat[i] != 0)
		{
			CSVOColor colorPacked;
			CVoxelNorm normalPacked;
			UnpackSVOMaterial(colorPacked, normalPacked, mat[i]);
			float4 color = UnpackSVOColor(colorPacked);
			float3 normal = ExpandOnlyNormal(normalPacked);

			colorAvg.x += color.x;
			colorAvg.y += color.y;
			colorAvg.z += color.z;

			normalAvg.x += normal.x;
			normalAvg.y += normal.y;
			normalAvg.z += normal.z;

			count++;
		}
	}

	if(threadIdx.x % 4 &&
	   currentLevel >= svoConstants.totalDepth)
	{
		// Parent also may contain color fetch and add it to average
		CSVOColor colorPacked;
		CVoxelNorm normalPacked;
		UnpackSVOMaterial(colorPacked, normalPacked, gSVOMat[matOffset + globalParentId]);
		float4 color = UnpackSVOColor(colorPacked);
		float3 normal = ExpandOnlyNormal(normalPacked);

		colorAvg.x += color.x;
		colorAvg.y += color.y;
		colorAvg.z += color.z;

		normalAvg.x += normal.x;
		normalAvg.y += normal.y;
		normalAvg.z += normal.z;

		// Wieghted average since this color spans more area (8 times more)
		count += 8;
	}

	// Average Between Threads
	for(int offset = 4 / 2; offset > 0; offset /= 2)
	{
		colorAvg.x += __shfl_down(colorAvg.x, offset, 4);
		colorAvg.y += __shfl_down(colorAvg.y, offset, 4);
		colorAvg.z += __shfl_down(colorAvg.z, offset, 4);

		normalAvg.x += __shfl_down(normalAvg.x, offset, 4);
		normalAvg.y += __shfl_down(normalAvg.y, offset, 4);
		normalAvg.z += __shfl_down(normalAvg.z, offset, 4);

		count += __shfl_down(count, offset, 4);
	}

	if(threadIdx.x % 4 == 0)
	{
		// Parent Thread Writes to smem (for coalesced write)
		float countInv = 1.0f / static_cast<float>(count);

		colorAvg.x *= countInv;
		colorAvg.y *= countInv;
		colorAvg.z *= countInv;

		normalAvg.x *= countInv;
		normalAvg.y *= countInv;
		normalAvg.z *= countInv;

		colorAvg.w = static_cast<unsigned char>(count);
		sMatAvg[threadIdx.x / 4] = PackSVOMaterial(PackSVOColor(colorAvg),
												   PackOnlyVoxNorm(normalAvg));

	}
	__syncthreads();

	// Write back
	if(threadIdx.x < blockDim.x / 4)
	{
		if(currentLevel > svoConstants.denseDepth)
			gSVOMat[matOffset + sNodeIds[threadIdx.x]] = sMatAvg[threadIdx.x];
		else
		{
			// Dense Write
			// Calculate Dense Start Location
			unsigned int lvlOffset = static_cast<unsigned int>((1.0 - powf(8.0f, currentLevel + 1)) / (1.0f - 8.0f));
			gSVOMat[lvlOffset + linearParentId] = sMatAvg[threadIdx.x];
		}
	}
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
									  svoConstants.numCascades,
									  svoConstants.totalDepth);

	unsigned int location;
	unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
	for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
	{
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

		// Offset child
		unsigned int childId = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
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

	//gSVOMat[matSparseOffset + location] = PackSVOMaterial(voxelColorPacked,
	//													  voxelNormPacked);
}