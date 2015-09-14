#include "CHash.cuh"
#include "CMatrix.cuh"
#include "CVoxelPage.h"

// Nearest bigger prime of "power of twos"
// This will be used when VS have full constexpr support
// Then we'll lookup this determine hash size in compile time
// so that changing segment time will automatically change GI_SEGMENT_PER_PAGE_PRIME value
//__device__ static const uint1 nearestPrimeofTwo[] =
//{
//	2,		// 2
//	5,		// 4
//	11,		// 8
//	17,		// 16
//	37,		// 32
//	67,		// 64,
//	131,	// 128
//	257		// 256,
//};

// Assertion Check
static_assert((GI_SEGMENT_PER_PAGE + 3) * (sizeof(CMatrix4x4) * 8 + sizeof(uint1)) <= GI_CUDA_SHARED_MEM_SIZE, "Not Enough Shared Mem to Store Hashed Transforms");
static_assert(GI_MAX_SHARED_COUNT_PRIME - GI_MAX_SHARED_COUNT <= 3 &&
			  GI_MAX_SHARED_COUNT_PRIME - GI_MAX_SHARED_COUNT > 0, "Shared count and its prime value does not seem to be related");

// Helper Functions
__device__ unsigned int MergeObjId(const ushort2& objectId)
{
	unsigned int result = 0;
	result |= static_cast<unsigned int>(objectId.y) << 16;
	result |= static_cast<unsigned int>(objectId.x);
	return result;
}

__device__ unsigned int Map(const ushort2& objectId, unsigned int* sHashIndex)
{
	unsigned int mergedObjId = MergeObjId(objectId);
	
	// CAS Loop to atomically find location of the merged objId
	// Hash table resolves collisions linearly
	// InitialIndex
	unsigned int index = (mergedObjId % GI_SEGMENT_PER_PAGE) - 1;
	unsigned int old = 0;
	do
	{
		index++;
		index %= GI_MAX_SHARED_COUNT_PRIME;
		old = atomicCAS(sHashIndex + index, 0, mergedObjId);
	} while(old != 0 && old != mergedObjId);

	// Worst case there is GI_MAX_SHARED_COUNT_PRIME elements in the hash
	// and our alloc is nearest prime which gurantees intex table has empty spaces
	// Thus we dont need to check that the lookup goes to a infinite loop
	return index;
}