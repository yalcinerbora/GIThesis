#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CHash.cuh"

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

inline __device__ unsigned char CalculateLevelChildId(const uint3& voxelPos,
													  const unsigned int totalDepth,
													  const unsigned int levelDepth)
{
	unsigned int bitSet = 0;
	bitSet = ((voxelPos.z >> (totalDepth - levelDepth - 1)) & 0x000000001) << 2;
	bitSet |= ((voxelPos.y >> (totalDepth - levelDepth - 1)) & 0x000000001) << 1;
	bitSet |= ((voxelPos.x >> (totalDepth - levelDepth - 1)) & 0x000000001) << 0;
	return static_cast<unsigned char>(bitSet);
}

inline __device__ uint3 CalculateLevelVoxId(const uint3& voxelPos,
											const unsigned int totalDepth,
											const unsigned int levelDepth)
{
	uint3 levelVoxelId;
	levelVoxelId.x = (voxelPos.x >> (totalDepth - levelDepth));
	levelVoxelId.y = (voxelPos.y >> (totalDepth - levelDepth));
	levelVoxelId.z = (voxelPos.z >> (totalDepth - levelDepth));
	return levelVoxelId;
}

inline __device__ uint3 UnpackLevelVoxId(unsigned int packedVoxel,
										 const unsigned int levelDepth)
{
	uint3 levelVoxelId;
	levelVoxelId.x = (((0x00000001 << levelDepth * 2) - 1) & packedVoxel) >> (levelDepth * 0);
	levelVoxelId.y = (((0x00000001 << levelDepth * 1) - 1) & packedVoxel) >> (levelDepth * 1);
	levelVoxelId.z = (((0x00000001 << levelDepth * 0) - 1) & packedVoxel) >> (levelDepth * 2);
	return levelVoxelId;
}

inline __device__ uint3 UnpackDenseVoxId(unsigned int packedVoxel,
										 const unsigned int levelDepth,
										 const unsigned int denseDepth)
{
	assert(denseDepth < levelDepth);
	uint3 denseVoxelId;
	denseVoxelId.x = (((0x00000001 << levelDepth * 2) - 1) & packedVoxel) >> (levelDepth * 0);
	denseVoxelId.y = (((0x00000001 << levelDepth * 1) - 1) & packedVoxel) >> (levelDepth * 1);
	denseVoxelId.z = (((0x00000001 << levelDepth * 0) - 1) & packedVoxel) >> (levelDepth * 2);

	denseVoxelId.x = denseVoxelId.x >> (levelDepth - denseDepth);
	denseVoxelId.y = denseVoxelId.y >> (levelDepth - denseDepth);
	denseVoxelId.z = denseVoxelId.z >> (levelDepth - denseDepth);
	return denseVoxelId;
}

inline __device__ void CalculateLevelData(unsigned int& packedVoxLevel,
										  unsigned char& childBit,
										  const uint3& voxelPos,
										  const unsigned int totalDepth,
										  const unsigned int levelDepth)
{
	// Child bit for the next level
	childBit = CalculateLevelChildId(voxelPos, totalDepth, levelDepth);
	childBit = 0x00000001 << (childBit - 1);

	// Voxel Level Id
	uint3 levelVoxelId = CalculateLevelVoxId(voxelPos, totalDepth, levelDepth);

	// Packed Vox Level That used for hashing
	packedVoxLevel = voxelPos.z << (levelDepth * 2);
	packedVoxLevel |= voxelPos.y << (levelDepth * 1);
	packedVoxLevel |= voxelPos.x << (levelDepth * 0);
}

// Low Populates Texture
__global__ void SVOReconstructChildSet(CSVONode* gSVODense,
									   const CVoxelPage* gVoxelData,
									   const unsigned int denseDim,
									   const unsigned int denseDepth,
									   const unsigned int totalDepth)
{
	__shared__ unsigned int sLocationHash[GI_THREAD_PER_BLOCK_PRIME];
	__shared__ CSVONode sNode[GI_THREAD_PER_BLOCK_PRIME];

	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Init Hashtable
	sLocationHash[threadIdx.x] = 0;
	if(GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK)
		sLocationHash[GI_THREAD_PER_BLOCK + threadIdx.x] = 0;
	__syncthreads();

	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];

	// Skip voxel if invalid
	if(voxelNormPos.y != 0xFFFFFFFF)
	{
		// Hash this levels locations in shared
		// Hash locations are same
		uint3 voxelPos = ExpandOnlyVoxPos(voxelNormPos.x);
		unsigned char childBit;
		unsigned int packedVoxLevel;

		CalculateLevelData(packedVoxLevel,
						   childBit,
						   ExpandOnlyVoxPos(voxelNormPos.x),
						   totalDepth,
						   denseDepth);

		// Atomic Hash Location find and write
		unsigned int  location = Map(sLocationHash, packedVoxLevel, GI_THREAD_PER_BLOCK_PRIME);
		atomicOr(sNode + location, PackNode(0, static_cast<unsigned char>(childBit)));
	}

	// Wait everything to be written
	__syncthreads();

	// Kill unwritten table indices
	if(sNode[threadIdx.x] == 0) return;

	// Thread logic changes
	// Write the stored tree node in shared mem
	// Global write to denseVoxel Array
	uint3 levelIndex = UnpackLevelVoxId(sLocationHash[threadIdx.x], denseDepth);
	atomicOr(gSVODense +
			 denseDim * denseDim * levelIndex.z +
			 denseDim * levelIndex.y +
			 levelIndex.x,
			 sNode[threadIdx.x]);
}

__global__ void SVOReconstructChildSet(CSVONode* gSVOSparse,
									   const CSVONode* gSVODense,
									   const CVoxelPage* gVoxelData,
									   const unsigned int* gLevelLookupTable,
									   const unsigned int levelDepth,
									   const unsigned int denseDepth,
									   const unsigned int totalDepth,
									   const unsigned int denseDim)
{
	__shared__ unsigned int sLocationHash[GI_THREAD_PER_BLOCK_PRIME];
	__shared__ CSVONode sNode[GI_THREAD_PER_BLOCK_PRIME];

	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Init Hashtable
	sLocationHash[threadIdx.x] = 0;
	if(GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK)
		sLocationHash[GI_THREAD_PER_BLOCK + threadIdx.x] = 0;
	__syncthreads();

	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];

	// Skip voxel if invalid
	uint3 voxelPos;
	if(voxelNormPos.y != 0xFFFFFFFF)
	{
		// Hash this levels locations in shared
		// Hash locations are same
		voxelPos = ExpandOnlyVoxPos(voxelNormPos.x);
		unsigned char childBit;
		unsigned int packedVoxLevel;

		CalculateLevelData(packedVoxLevel,
						   childBit,
						   voxelPos,
						   totalDepth,
						   levelDepth);

		// Atomic Hash Location find and write
		unsigned int  location = Map(sLocationHash, packedVoxLevel, GI_THREAD_PER_BLOCK_PRIME);
		atomicOr(sNode + location, PackNode(0, static_cast<unsigned char>(childBit)));
	}

	// Wait everything to be written
	__syncthreads();

	// Kill unwritten table threads 
	// TODO: Branch divergenge should be high because of hash table uniformness
	// try to reduce
	if(sNode[threadIdx.x] == 0) return;

	// Global write to denseVoxel Array
	// Find the child
	uint3 denseIndex = UnpackDenseVoxId(sLocationHash[threadIdx.x], levelDepth, denseDepth);
	CSVONode currentNode = gSVODense[denseDim * denseDim * denseIndex.z +
									 denseDim * denseIndex.y +
									 denseIndex.z];
	unsigned int nodeIndex = 0;
	for(unsigned int i = denseDepth + 1; i <= levelDepth; i++)
	{
		unsigned int levelBase = gLevelLookupTable[i];
		
		unsigned char childBits;
		unsigned int childIndex;
		UnpackNode(childIndex, childBits, currentNode);

		// Determine CurrentLevels Child
		unsigned char childId = CalculateLevelChildId(voxelPos, totalDepth, i);
		nodeIndex = levelBase + childIndex + childId;
		currentNode = gSVOSparse[nodeIndex];
	}

	// Finally Write
	atomicOr(gSVOSparse + nodeIndex, sNode[threadIdx.x]);
}


__global__ void SVOReconstructAllocateNext(CSVONode* gSVOLevel,
										   unsigned int& gSVOLevelNodeCount,
										   const unsigned int& gSVOLevelStart,
										   const unsigned levelDim)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= levelDim * levelDim * levelDim) return;

	CSVONode node = gSVOLevel[gSVOLevelStart + globalId];
	unsigned int childCount;
	unsigned char childBits;
	UnpackNode(childCount, childBits, node);
	
	childCount = 0;
	#pragma unroll 
	for(unsigned int i = 0; i < 8; i++)
	{
		childCount += childBits >> i & 0x01;
	}

	// Allocation
	unsigned int location = atomicAdd(&gSVOLevelNodeCount, childCount);
	assert(location <= 0x00FFFFFF);
	gSVOLevel[gSVOLevelStart + globalId] = PackNode(location, childBits);
}						   