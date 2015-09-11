#include "CDebug.cuh"
#include "CMatrix.cuh"
#include "CAxisAlignedBB.cuh"
#include "CVoxel.cuh"
#include "COpenGLCommon.cuh"
#include <cassert>
#include <limits>

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
	bool intersects = CheckGridVoxIntersect(gGridInfo, gObjectAABB[objectId], gObjTransforms[objectId]);
	ushort2 myAllocLoc = gObjectAllocLocations[globalId];

	if(intersects)
	{
		assert(myAllocLoc.x != 0xFFFF &&
			   myAllocLoc.y != 0xFFFF);
	}
	else
	{
		assert(myAllocLoc.x == 0xFFFF &&
			   myAllocLoc.y == 0xFFFF);
	}
}


