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

__global__ void SVOReconstructChildSet(CSVONode* gSVODense,
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
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];
	if(voxelNormPos.y == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelPos = ExpandToSVODepth(ExpandOnlyVoxPos(voxelNormPos.x),
									  cascadeNo,
									  svoConstants.numCascades);
	unsigned char childBit = CalculateLevelChildBit(voxelPos,
													svoConstants.denseDepth + 1,
													svoConstants.totalDepth);
	uint3 levelVoxId = CalculateLevelVoxId(voxelPos,
										   svoConstants.denseDepth,
										   svoConstants.totalDepth);
	
	assert(levelVoxId.x < svoConstants.denseDim &&
		   levelVoxId.y < svoConstants.denseDim &&
		   levelVoxId.z < svoConstants.denseDim);

	// Actual child bit set
	atomicOr(gSVODense +
			 svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
			 svoConstants.denseDim * levelVoxId.y +
			 levelVoxId.x,
			 PackNode(0, static_cast<unsigned char>(childBit)));

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
	unsigned int globalId = threadIdx.x + blockIdx.x * GI_THREAD_PER_BLOCK;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];

	if(voxelNormPos.y == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelPos = ExpandToSVODepth(ExpandOnlyVoxPos(voxelNormPos.x),
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

	//if(levelDepth == 9)

	assert(denseIndex.x < svoConstants.denseDim &&
		   denseIndex.y < svoConstants.denseDim &&
		   denseIndex.z < svoConstants.denseDim);

	CSVONode currentNode = tex3D<unsigned int>(tSVODense,
											   denseIndex.x,
											   denseIndex.y,
											   denseIndex.z);
	
	//if(currentNode == 0)
	//{
	//	printf("Assert DenseMissXYZ 0x%X, 0x%X, 0x%X\n",
	//	denseIndex.x,
	//	denseIndex.y,
	//	denseIndex.z);
	//}

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
		unsigned char childIndex = CalculateChildIndex(childBits, requestedChild, i);

		nodeIndex = levelBase + childrenStart + childIndex;

		// Last gMem read unnecessary
		if(i < levelDepth) currentNode = gSVOSparse[nodeIndex];
	}

	// Finally Write
	atomicOr(gSVOSparse + nodeIndex, PackNode(0, static_cast<unsigned char>(childBit)));
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

__global__ void SVOReconstructAverageNode(cudaTextureObject_t tSVODense,
										  const CVoxelPage* gVoxelData,
										  CSVOMaterial* material
										  )
{

}