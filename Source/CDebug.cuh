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

// Debug Kernel that checks that we allocated unqiuely
// Only works if there is a single batch in system
// Call logic per segment
__global__ void DebugCheckUniqueAlloc(ushort2* gObjectAllocLocations,
									  unsigned int segmentCount);

// Checks if the interected object has all of its segments allocated
// this requires very big page system so that objects guaranteed to have
// its segments fully allocated
// Call logic per segment
__global__ void DebugCheckSegmentAlloc(const CVoxelGrid& gGridInfo,

									   const ushort2* gObjectAllocLocations,
									   const unsigned int* gSegmentObjectId,
									   unsigned int segmentCount,

									   const CObjectAABB* gObjectAABB,
									   const CObjectTransform* gObjTransforms);

#endif //__CDEBUG_H__