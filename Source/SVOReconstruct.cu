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

inline __device__ void StoreDense(CSVONode* gSVODense,
								  const unsigned int* sLocationHash,
								  const CSVONode* sNode, 
								  unsigned int sharedLoc,
								  unsigned int cascadeNo, 
								  const CSVOConstants& svoConstants)
{
	// Kill unwritten table indices
	if(sLocationHash[sharedLoc] == 0xFFFFFFFF) return;
	assert(sNode[sharedLoc] != 0x00000000);

	// Thread logic changes
	// Write the stored tree node in shared mem
	// Global write to denseVoxel Array
	uint3 levelIndex = KeyToPos(sLocationHash[sharedLoc],
								cascadeNo,
								svoConstants.denseDepth,
								svoConstants.numCascades);

	assert(levelIndex.x < svoConstants.denseDim &&
		   levelIndex.y < svoConstants.denseDim &&
		   levelIndex.z < svoConstants.denseDim);

	// Actual child bit set
	atomicOr(gSVODense +
			 svoConstants.denseDim * svoConstants.denseDim * levelIndex.z +
			 svoConstants.denseDim * levelIndex.y +
			 levelIndex.x,
			 sNode[sharedLoc]);
}

inline __device__ void StoreSparse(CSVONode* gSVOSparse,
								   const unsigned int* gLevelLookupTable,
								   const unsigned int* sLocationHash,
								   const CSVONode* sNode,
								   cudaTextureObject_t tSVODense,

								   unsigned int sharedLoc,
								   unsigned int cascadeNo,
								   unsigned int levelDepth,
								   const CSVOConstants& svoConstants)
{
	// Kill unwritten table threads 
	if(sLocationHash[sharedLoc] == 0xFFFFFFFF) return;
	assert(sNode[sharedLoc] != 0x00000000);

	// Thread logic changes
	// Traverse the partially constructed tree and put the child
	uint3 levelVoxelId = KeyToPos(sLocationHash[sharedLoc],
								  cascadeNo,
								  levelDepth,
								  svoConstants.numCascades);
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
	//if(currentNode == 0)
	//{
	//	printf("Assert DenseMissXYZ 0x%X, 0x%X, 0x%X\n",
	//	denseIndex.x,
	//	denseIndex.y,
	//	denseIndex.z);
	//}


	unsigned int nodeIndex = 0;
	for(unsigned int i = svoConstants.denseDepth + 1; i <= levelDepth; i++)
	{
		/*volatile*/ unsigned int levelBase = gLevelLookupTable[i - svoConstants.denseDepth - 1];

		unsigned char childBits;
		unsigned int childrenStart;
		UnpackNode(childrenStart, childBits, currentNode);
		assert(childBits != 0);

		// Jump to Next Node
		/*volatile*/ unsigned char requestedChild = CalculateLevelChildBit(levelVoxelId, i, levelDepth);
		/*volatile*/ unsigned char childIndex = CalculateChildIndex(childBits, requestedChild, i);

		nodeIndex = levelBase + childrenStart + childIndex;

		// Last gMem read unnecessary
		if(i < levelDepth) currentNode = gSVOSparse[nodeIndex];
	}

	// Finally Write
	atomicOr(gSVOSparse + nodeIndex, sNode[sharedLoc]);
}

inline __device__ void HashStoreLevel(CSVONode* sNode,
									  unsigned int* sLocationHash,
									  const CVoxelNormPos& voxelNormPos,
									  const unsigned int level,
									  const unsigned int cascadeNo,
									  const CSVOConstants& svoConstants)
{
	// Skip voxel if invalid
	if(voxelNormPos.y != 0xFFFFFFFF)
	{
		// Local Voxel pos and expand it if its one of the inner cascades
		uint3 voxelPos = ExpandToSVODepth(ExpandOnlyVoxPos(voxelNormPos.x),
										  cascadeNo,
										  svoConstants.numCascades);
		/*volatile*/ unsigned char childBit = CalculateLevelChildBit(voxelPos,
																 level + 1,
																 svoConstants.totalDepth);
		uint3 levelVoxId = CalculateLevelVoxId(voxelPos,
											   level,
											   svoConstants.totalDepth);

		/*volatile*/ unsigned int packedVoxLevel = PosToKey(levelVoxId, level);

		// Atomic Hash Location find and write
		//unsigned int location = packedVoxLevel % GI_THREAD_PER_BLOCK_PRIME;
		//sLocationHash[location] = packedVoxLevel;

		//unsigned int old = atomicCAS(&sLocationHash[location], 0xFFFFFFFF, packedVoxLevel);
		//sLocationHash[location] = packedVoxLevel;

		unsigned int  location = Map(sLocationHash, packedVoxLevel, GI_THREAD_PER_BLOCK_PRIME);
		atomicOr(sNode + location, PackNode(0, static_cast<unsigned char>(childBit)));
	}
}

__global__ void SVOReconstructChildSet(CSVONode* gSVODense,
									   const CVoxelPage* gVoxelData,

									   const unsigned int cascadeNo,
									   const CSVOConstants& svoConstants)
{
	__shared__ unsigned int sLocationHash[GI_THREAD_PER_BLOCK_PRIME];
	__shared__ CSVONode sNode[GI_THREAD_PER_BLOCK_PRIME];

	unsigned int globalId = threadIdx.x + blockIdx.x * GI_THREAD_PER_BLOCK;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Init Hashtable
	sLocationHash[threadIdx.x] = 0xFFFFFFFF;
	sNode[threadIdx.x] = 0;
	if(threadIdx.x < (GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK))
	{
		sLocationHash[threadIdx.x + GI_THREAD_PER_BLOCK] = 0xFFFFFFFF;
		sNode[threadIdx.x + GI_THREAD_PER_BLOCK] = 0;
	}
	__syncthreads();

	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];
	HashStoreLevel(sNode,
					sLocationHash,
					voxelNormPos,
					svoConstants.denseDepth,
					cascadeNo,
					svoConstants);

	// Wait everything to be written
	__syncthreads();

	StoreDense(gSVODense,
			   sLocationHash,
			   sNode,
			   threadIdx.x,
			   cascadeNo,
			   svoConstants);
	if(threadIdx.x < (GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK))
	{
		StoreDense(gSVODense,
				   sLocationHash,
				   sNode,
				   threadIdx.x + GI_THREAD_PER_BLOCK,
				   cascadeNo,
				   svoConstants);
	}

	////DEBUG
	//if(blockIdx.x == 0)
	//{
	//	for(unsigned int wid = 0; wid < blockDim.x / 32; wid++)
	//	{
	//		if(threadIdx.x / 32 == wid &&
	//		   threadIdx.x < GI_THREAD_PER_BLOCK_PRIME)
	//			printf("#%d{0x%X, 0x%X}\n", threadIdx.x, sNode[threadIdx.x], sLocationHash[threadIdx.x]);
	//		__syncthreads();
	//	}
	//}
}

__global__ void SVOReconstructChildSet(CSVONode* gSVOSparse,
									   cudaTextureObject_t tSVODense,
									   const CVoxelPage* gVoxelData,
									   const unsigned int* gLevelLookupTable,

									   // Constants
									   const unsigned int cascadeNo,
									   const unsigned int levelDepth,
									   const CSVOConstants& svoConstants)
{
	__shared__ unsigned int sLocationHash[GI_THREAD_PER_BLOCK_PRIME];
	__shared__ CSVONode sNode[GI_THREAD_PER_BLOCK_PRIME];

	unsigned int globalId = threadIdx.x + blockIdx.x * GI_THREAD_PER_BLOCK;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Init Hashtable
	sLocationHash[threadIdx.x] = 0xFFFFFFFF;
	sNode[threadIdx.x] = 0;
	if(threadIdx.x < (GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK))
	{
		sLocationHash[threadIdx.x + GI_THREAD_PER_BLOCK] = 0xFFFFFFFF;
		sNode[threadIdx.x + GI_THREAD_PER_BLOCK] = 0;
	}
	__syncthreads();

	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];
	HashStoreLevel(sNode,
					sLocationHash,
					voxelNormPos,
					levelDepth,
					cascadeNo,
					svoConstants);

	// Wait everything to be written
	__syncthreads();

	StoreSparse(gSVOSparse, gLevelLookupTable,
				sLocationHash, sNode,
				tSVODense, threadIdx.x,
				cascadeNo, levelDepth,
				svoConstants);
	if(threadIdx.x < (GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK))
	{
		StoreSparse(gSVOSparse, gLevelLookupTable,
					sLocationHash, sNode,
					tSVODense, threadIdx.x + GI_THREAD_PER_BLOCK,
					cascadeNo, levelDepth,
					svoConstants);
	}
}

__global__ void SVOReconstructAllocateNext(CSVONode* gSVO,
										   unsigned int& gSVOLevelNodeCount,
										   const unsigned int& gSVOLevelOffset,
										   const unsigned int& gSVONextLevelOffset,
										   const unsigned levelSize)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= levelSize) return;

	CSVONode node = gSVO[globalId + gSVOLevelOffset];
	if(node == 0x00000000) return;

	unsigned int childCount;
	unsigned char childBits;
	UnpackNode(childCount, childBits, node);	
	childCount = __popc(childBits);
	
	// Allocation
	unsigned int location = atomicAdd(&gSVOLevelNodeCount, childCount);
	unsigned int localLocation = location - gSVONextLevelOffset;
	assert(localLocation <= 0x00FFFFFF);

	gSVO[globalId + gSVOLevelOffset] = PackNode(localLocation, childBits);
}						   