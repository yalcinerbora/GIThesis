#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CVoxel.cuh"

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

										  const CSVONode* gSVOSparse,

										  const unsigned int matOffset,
										  const unsigned int svoLevelOffset,

										  const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int localNodeId = threadIdx.x / 4;
	unsigned int localNodeChildId = threadIdx.x % 4;
	
	__shared__ CSVONode sNodeIds[GI_THREAD_PER_BLOCK / 4];

	// coalesced access
	if(threadIdx.x < blockDim.x / 4)
	{
		sNodeIds[threadIdx.x] = gSVOSparse[svoLevelOffset + globalId];
	}
	__syncthreads();

	// Load Children
	CSVONode nodeLow, nodeHi;
	unsigned int nodeId = sNodeIds[localNodeId] + localNodeChildId;
	nodeLow = (sNodeIds[localNodeId] != 0xFFFFFFFF) ? gSVOSparse[nodeId] : 0xFFFFFFFF;
	nodeHi = (sNodeIds[localNodeId] != 0xFFFFFFFF) ? gSVOSparse[nodeId + 4] : 0xFFFFFFFF;
	// Each Thread has two children
	// T1 -> 0, 4
	// T2 -> 1, 5
	// T3 -> 2, 6
	// T4 -> 3, 7

	// Load Material
	CSVOMaterial matLow, matHi;
	matLow = (nodeLow != 0xFFFFFFFF) ? gSVOMat[matOffset + nodeLow] : 0;
	matHi = (nodeLow != 0xFFFFFFFF) ? gSVOMat[matOffset + nodeHi] : 0;

	//
	ushort4 color = {0, 0, 0, 0};
	float3 normal = {0.0f, 0.0f, 0.0f};

	//UnpackSVOMaterial(color, normal, matLow);



	//// Average Material
	//for(int offset = 4 / 2; offset > 0; offset /= 2)
	//{

	//	CVoxelNorm normal = __shfl_down(.x, offset, 4);
	//	CSVOColor color = __shfl_down(val.x, offset, 4);
	//}
	//// 

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