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
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelPos = ExpandToSVODepth(ExpandOnlyVoxPos(voxelPosPacked),
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

__global__ void SVOReconstructAverageLeaf(CSVOMaterial* gSVOMat,
										  
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

	// Actual Atomic Average
	AtomicColorNormalAvg(gSVOMat + nodeIndex + matSparseOffset, voxelColorPacked, voxelNormPacked);
}

extern __global__ void SVOReconstructAverageNode(CSVOMaterial* parentMats,
												 const CSVOMaterial* childrenMats,
												 const CSVONode* gSVONode,
												 const unsigned int parentLevel,
												 const CSVOConstants& svoConstants)
{

}