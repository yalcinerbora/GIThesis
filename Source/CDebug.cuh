/*

Debug Functions

*/

#ifndef __CDEBUG_H__
#define __CDEBUG_H__

#include <cuda.h>
#include <cuda_runtime.h>

struct CAABB;
typedef CAABB CObjectAABB;
struct CObjectTransform;
struct CVoxelGrid;
typedef unsigned int CSVONode;
struct CSVOConstants;

// Debug Kernel that checks that we allocated unqiuely
// Each segment compares its id with all other segments
// Only works if there is a single batch in system
// Call logic per segment
__global__ void DebugCheckUniqueAlloc(ushort2* gObjectAllocLocations,
									  unsigned int segmentCount);

// Checks if the interected object has all of its segments allocated
// Each segment does an intersection test and checks if it is allocated.
// this requires very big page system so that objects guaranteed to have
// its segments fully allocated
// Call logic per segment
__global__ void DebugCheckSegmentAlloc(const CVoxelGrid& gGridInfo,

									   const ushort2* gObjectAllocLocations,
									   const unsigned int* gSegmentObjectId,
									   unsigned int segmentCount,

									   const CObjectAABB* gObjectAABB,
									   const CObjectTransform* gObjTransforms);

// Checks if the written Node Id can be used to traverse the pointing node
extern __global__ void DebugCheckNodeId(const CSVONode* gSVODense,
										const CSVONode* gSVOSparse,

										const unsigned int* gNodeIds,
										const unsigned int* gSVOLevelOffsets,
										const unsigned int& gSVOLevelOffset,
										const unsigned int levelNodeCount,
										const unsigned int currentLevel,
										const CSVOConstants& svoConstants);

#endif //__CDEBUG_H__