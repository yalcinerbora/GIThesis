#include "CDebug.cuh"
#include "CMatrix.cuh"
#include "CAxisAlignedBB.cuh"
#include "CVoxel.cuh"
#include "CSVOTypes.cuh"
#include "CSparseVoxelOctree.cuh"
#include <cassert>
#include <limits>
#include <cstdio>

__global__ void DebugCheckNodeId(const CSVONode* gSVODense,
								 const CSVONode* gSVOSparse,
								 const unsigned int* gNodeIds,
								 const unsigned int* gSVOLevelOffsets,
								 const unsigned int& gSVOLevelOffset,
								 const unsigned int levelNodeCount,
								 const unsigned int currentLevel,
								 const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	// Cull if out of range
	if(globalId > levelNodeCount) return;

	// Read Sibling Materials
	const CSVONode* n = (currentLevel == svoConstants.denseDepth) ? gSVODense : gSVOSparse;
	CSVONode node = n[gSVOLevelOffset + globalId];

	// Cull if there is no children
	if(node == 0xFFFFFFFF) return;

	// Read Node Adress (we will compare this)
	const unsigned int nodeId = gNodeIds[gSVOLevelOffset + globalId];
	uint3 voxUnexpanded = ExpandOnlyVoxPos(nodeId);
	uint3 voxNode = ExpandNodeIdToDepth(voxUnexpanded, currentLevel,
										svoConstants.numCascades,
										svoConstants.totalDepth);

	// Traverse SVO
	uint3 levelVoxId = CalculateLevelVoxId(voxNode, svoConstants.denseDepth, svoConstants.totalDepth);
	CSVONode location = gSVODense[svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
								  svoConstants.denseDim * levelVoxId.y +
								  levelVoxId.x];

	if(location == 0xFFFFFFFF)
	{
		printf("voxelNode : %d, %d, %d\n"
			   "voxUnpacked : %d, %d, %d\n"
			   "checkId : %d, %d, %d\n"
			   "level: %d\n"
			   "notFoundOn: %d\n"
			   "------------\n",
			   voxNode.x, voxNode.y, voxNode.z,
			   voxUnexpanded.x, voxUnexpanded.y, voxUnexpanded.z,
			   levelVoxId.x, levelVoxId.y, levelVoxId.z,
			   currentLevel, svoConstants.denseDepth);
		assert(false);
	}

	location += CalculateLevelChildId(voxNode, svoConstants.denseDepth + 1, svoConstants.totalDepth);
	for(unsigned int i = svoConstants.denseDepth + 1; i < currentLevel; i++)
	{
		unsigned int levelIndex = i - svoConstants.denseDepth;
		const CSVONode node = gSVOSparse[gSVOLevelOffsets[levelIndex] + location];		
		//assert(location != 0xFFFFFFFF);
		if(node == 0xFFFFFFFF)
		{
			uint3 checkId = CalculateLevelVoxId(voxNode, i, svoConstants.totalDepth);

			printf("voxelNode : %d, %d, %d\n"
			"voxUnpacked : %d, %d, %d\n"
			"checkId : %d, %d, %d\n"
			"level: %d\n"
			"notFoundOn: %d\n"
			"------------\n",
			voxNode.x, voxNode.y, voxNode.z,
			voxUnexpanded.x, voxUnexpanded.y, voxUnexpanded.z,
			checkId.x, checkId.y, checkId.z,
			currentLevel, i);
			assert(false);
		}

		// Offset child
		unsigned int childId = CalculateLevelChildId(voxNode, i + 1, svoConstants.totalDepth);
		location = node + childId;
	}

	// Assertion
	unsigned int levelIndex = currentLevel - svoConstants.denseDepth;
	assert(globalId == gSVOSparse[gSVOLevelOffsets[levelIndex] + location]);

}

__global__ void DebugCheckUniqueAlloc(ushort2* gObjectAllocLocations,
									  unsigned int segmentCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= segmentCount) return;

	ushort2 myAllocLoc = gObjectAllocLocations[globalId];
	for(unsigned int i = 0; i < segmentCount; i++)
	{
		ushort2 otherSegment = gObjectAllocLocations[i];
		if(i != globalId &&
		   myAllocLoc.x != 0xFFFF &&
		   myAllocLoc.y != 0xFFFF &&
		   myAllocLoc.x == otherSegment.x &&
		   myAllocLoc.y == otherSegment.y)
		{
			assert(false);
		}
	}
}

__global__ void DebugCheckSegmentAlloc(const CVoxelGrid& gGridInfo,

									   const ushort2* gObjectAllocLocations,
									   const unsigned int* gSegmentObjectId,
									   unsigned int segmentCount,

									   const CObjectAABB* gObjectAABB,
									   const CObjectTransform* gObjTransforms)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= segmentCount) return;

	unsigned int objectId = gSegmentObjectId[globalId];
	bool intersects = CheckGridVoxIntersect(gGridInfo, gObjectAABB[objectId], gObjTransforms[objectId].transform);
	ushort2 myAllocLoc = gObjectAllocLocations[globalId];
	
	if(intersects &&
	   (myAllocLoc.x == 0xFFFF ||
	   myAllocLoc.y == 0xFFFF))
	{
		assert(false);
	}

	if((!intersects) &&
	   (myAllocLoc.x != 0xFFFF ||
	   myAllocLoc.y != 0xFFFF))
	{
		assert(false);
	}

}


