//#include "CDebug.cuh"
//#include "CMatrixFunctions.cuh"
//#include "CAABBFunctions.cuh"
//#include "CVoxelFunctions.cuh"
//#include "CSVOTypes.h"
//#include "CSVOFunctions.cuh"
//#include <cassert>
//#include <limits>
//#include <cstdio>
//#include "COpenglTypes.h"
//
//__global__ void DebugCheckNodeId(const CSVONode* gSVODense,
//								 const CSVONode* gSVOSparse,
//								 const unsigned int* gNodeIds,
//								 const unsigned int* gSVOLevelOffsets,
//								 const unsigned int& gSVOLevelOffset,
//								 const unsigned int levelNodeCount,
//								 const unsigned int currentLevel,
//								 const CSVOConstants& svoConstants)
//{
//	const unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//
//	// Cull if out of range
//	if(globalId > levelNodeCount) return;
//	
//	// Read Sibling Materials
//	const CSVONode node = gSVOSparse[gSVOLevelOffset + globalId];
//
//	// Cull if there is no children
//	if(node == 0xFFFFFFFF) return;
//
//	// Read Node Adress (we will compare this)
//	const unsigned int nodeId = gNodeIds[gSVOLevelOffset + globalId];
//
//	uint3 voxNode = UnpackNodeId(nodeId, currentLevel,
//								 svoConstants.numCascades,
//								 svoConstants.totalDepth);
//
//	voxNode.x <<= (svoConstants.totalDepth - currentLevel);
//	voxNode.y <<= (svoConstants.totalDepth - currentLevel);
//	voxNode.z <<= (svoConstants.totalDepth - currentLevel);
//
//	// Traverse SVO
//	uint3 levelVoxId = CalculateLevelVoxId(voxNode, svoConstants.denseDepth, svoConstants.totalDepth);
//	CSVONode location = gSVODense[svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
//								  svoConstants.denseDim * levelVoxId.y +
//								  levelVoxId.x];
//	assert(location != 0xFFFFFFFF);
//	location += CalculateLevelChildId(voxNode, svoConstants.denseDepth + 1, svoConstants.totalDepth);
//	for(unsigned int i = svoConstants.denseDepth + 1; i <= currentLevel; i++)
//	{
//		unsigned int levelIndex = i - svoConstants.denseDepth;
//		const CSVONode node = gSVOSparse[gSVOLevelOffsets[levelIndex] + location];		
//		assert(node != 0xFFFFFFFF);
//
//		// Offset child
//		unsigned int childId = CalculateLevelChildId(voxNode, i + 1, svoConstants.totalDepth);
//		location = node + childId;
//	}
//}
//
//__global__ void DebugCheckUniqueAlloc(ushort2* gObjectAllocLocations,
//									  unsigned int segmentCount)
//{
//	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	if(globalId >= segmentCount) return;
//
//	ushort2 myAllocLoc = gObjectAllocLocations[globalId];
//	for(unsigned int i = 0; i < segmentCount; i++)
//	{
//		ushort2 otherSegment = gObjectAllocLocations[i];
//		if(i != globalId &&
//		   myAllocLoc.x != 0xFFFF &&
//		   myAllocLoc.y != 0xFFFF &&
//		   myAllocLoc.x == otherSegment.x &&
//		   myAllocLoc.y == otherSegment.y)
//		{
//			assert(false);
//		}
//	}
//}
//
//__global__ void DebugCheckSegmentAlloc(const CVoxelGrid& gGridInfo,
//
//									   const ushort2* gObjectAllocLocations,
//									   const unsigned int* gSegmentObjectId,
//									   unsigned int segmentCount,
//
//									   const CObjectAABB* gObjectAABB,
//									   const CObjectTransform* gObjTransforms)
//{
//	//unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	//if(globalId >= segmentCount) return;
//
//	//unsigned int objectId = gSegmentObjectId[globalId];
//	//bool intersects = CheckGridVoxIntersect(gGridInfo, 
//	//										gObjectAABB[objectId], 
//	//										gObjTransforms[objectId].transform);
//	//ushort2 myAllocLoc = gObjectAllocLocations[globalId];
//	//
//	//if(intersects &&
//	//   (myAllocLoc.x == 0xFFFF ||
//	//   myAllocLoc.y == 0xFFFF))
//	//{
//	//	assert(false);
//	//}
//
//	//if((!intersects) &&
//	//   (myAllocLoc.x != 0xFFFF ||
//	//   myAllocLoc.y != 0xFFFF))
//	//{
//	//	assert(false);
//	//}
//
//}
//
//
