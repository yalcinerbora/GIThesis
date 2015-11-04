#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CHash.cuh"

inline __device__ unsigned int CalculateChildIndex(volatile const unsigned char childrenBits,
												   volatile const unsigned char childBit)
{
	unsigned int childrenCount = __popc(childrenBits);
	unsigned int bit = childrenBits, totalBitCount = 0;
	for(unsigned int i = 0; i < childrenCount; i++)
	{
		totalBitCount += __ffs(bit);
		if((0x00000001 << totalBitCount - 1) == childBit)
			return i;
		bit = bit >> __ffs(bit);
		//totalBitCount++;
	}
	//assert(false);
	//return 0;

	
	unsigned int asd123 = childrenBits + childBit;
	printf("Assert childbit 0x%X, allbits 0x%X\n", childrenBits, childBit);
	assert(false);
	return asd123;
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
		unsigned char childBit = CalculateLevelChildBit(voxelPos,
														level + 1,
														svoConstants.totalDepth);
		uint3 levelVoxId = CalculateLevelVoxId(voxelPos,
											   level,
											   svoConstants.totalDepth);

		unsigned int packedVoxLevel = PosToKey(levelVoxId);

		// Atomic Hash Location find and write
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
	if(threadIdx.x < GI_THREAD_PER_BLOCK_PRIME)
	{
		sLocationHash[threadIdx.x] = 0xFFFFFFFF;
		sNode[threadIdx.x] = 0;
	}
	__syncthreads();

	if(threadIdx.x < GI_THREAD_PER_BLOCK)
	{
		// Fetch voxel
		CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];

		HashStoreLevel(sNode,
					   sLocationHash,
					   voxelNormPos,
					   svoConstants.denseDepth,
					   cascadeNo,
					   svoConstants);
	}

	// Wait everything to be written
	__syncthreads();

	if(threadIdx.x < GI_THREAD_PER_BLOCK_PRIME)
	{
		// Kill unwritten table indices
		if(sLocationHash[threadIdx.x] == 0xFFFFFFFF) return;
		assert(sNode[threadIdx.x] != 0x00000000);

		// Thread logic changes
		// Write the stored tree node in shared mem
		// Global write to denseVoxel Array
		uint3 levelIndex = KeyToPos(sLocationHash[threadIdx.x],
									cascadeNo,
									svoConstants.numCascades);

		assert(levelIndex.x < svoConstants.denseDim &&
			   levelIndex.y < svoConstants.denseDim &&
			   levelIndex.z < svoConstants.denseDim);

		// Actual child bit set
		atomicOr(gSVODense +
				 svoConstants.denseDim * svoConstants.denseDim * levelIndex.z +
				 svoConstants.denseDim * levelIndex.y +
				 levelIndex.x,
				 sNode[threadIdx.x]);
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
									   //const CSVONode* gSVODense,
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
	if(threadIdx.x < GI_THREAD_PER_BLOCK_PRIME)
	{
		sLocationHash[threadIdx.x] = 0xFFFFFFFF;
		sNode[threadIdx.x] = 0;
	}
	__syncthreads();

	if(threadIdx.x < GI_THREAD_PER_BLOCK)
	{
		// Fetch voxel
		CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];
		HashStoreLevel(sNode,
					   sLocationHash,
					   voxelNormPos,
					   levelDepth,
					   cascadeNo,
					   svoConstants);
	}

	// Wait everything to be written
	__syncthreads();

	if(threadIdx.x < GI_THREAD_PER_BLOCK_PRIME)
	{
		// Kill unwritten table threads 
		if(sLocationHash[threadIdx.x] == 0xFFFFFFFF) return;
		assert(sNode[threadIdx.x] != 0x00000000);

		// Thread logic changes
		// Traverse the partially constructed tree and put the child
		uint3 levelVoxelId = KeyToPos(sLocationHash[threadIdx.x],
									  cascadeNo,
									  svoConstants.numCascades);
		uint3 denseIndex = CalculateLevelVoxId(levelVoxelId, svoConstants.denseDepth,
											   levelDepth);

		assert(denseIndex.x < svoConstants.denseDim &&
			   denseIndex.y < svoConstants.denseDim &&
			   denseIndex.z < svoConstants.denseDim);

		/*CSVONode currentNode = gSVODense[svoConstants.denseDim * svoConstants.denseDim * denseIndex.z +
										 svoConstants.denseDim * denseIndex.y +
										 denseIndex.x];*/

		CSVONode currentNode = tex3D<unsigned int>(tSVODense, denseIndex.x,
												   denseIndex.y, denseIndex.z);


		unsigned int nodeIndex = 0;
		for(unsigned int i = svoConstants.denseDepth + 1; i <= levelDepth; i++)
		{
			volatile unsigned int levelBase = gLevelLookupTable[i - svoConstants.denseDepth - 1];

			unsigned char childBits;
			unsigned int childrenStart;
			UnpackNode(childrenStart, childBits, currentNode);
			
			// Jump to Next Node
			volatile unsigned char childIndex = CalculateChildIndex(childBits,
														   CalculateLevelChildBit(levelVoxelId,
														   i,
														   levelDepth));
			
			nodeIndex = levelBase + childrenStart + childIndex;


	/*		if(i == 3 &&
			   levelDepth == 3 &&
			   levelVoxelId.x == 0x02 &&
			   levelVoxelId.y == 0x02 &&
			   levelVoxelId.z == 0x02)
			{
				assert(false);
			}
			if(nodeIndex == 12)
			{
				assert(false);
			}*/

			// Last gMem read unnecessary
			if(i < levelDepth) currentNode = gSVOSparse[nodeIndex];

			/*if(currentNode == 0)
			{
				assert(currentNode != 0);
			}		*/		
		}

		// Finally Write
		atomicOr(gSVOSparse + nodeIndex, sNode[threadIdx.x]);
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