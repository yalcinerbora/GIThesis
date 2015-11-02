/*

Quickly Implemented Hashtable

operates on storing transform matrices into shared memory
in page system

*/

#include <cuda_runtime.h>
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

// Using shared memory on 16Kb config
#define GI_CUDA_SHARED_MEM_SIZE (16 * 1024)

// Max Worst case scenario for shared memory
#define GI_MAX_SHARED_COUNT 64
#define GI_MAX_SHARED_COUNT_PRIME 67

#define GI_THREAD_PER_BLOCK_PRIME 521

// Assertion Checks
static_assert((GI_MAX_SHARED_COUNT_PRIME) * (sizeof(CMatrix4x4) + sizeof(uint1)) <= GI_CUDA_SHARED_MEM_SIZE, "Not Enough Shared Mem to Store Hashed Transforms");
static_assert(GI_MAX_SHARED_COUNT_PRIME - GI_MAX_SHARED_COUNT == 3, "SharedCount and its prime does not seem to be related");
static_assert((GI_THREAD_PER_BLOCK_PRIME) * (2 * sizeof(uint1)) <= GI_CUDA_SHARED_MEM_SIZE, "Not Enough Shared Mem to Store Hashed Transforms");
static_assert(GI_THREAD_PER_BLOCK_PRIME - GI_THREAD_PER_BLOCK == 9, "ThreadPerBlock and its prime does not seem to be related");

// Mapping and "Allocating" the objectId
// Map functions manages memory conflicts internally thus can be called
// from different threads at the same time
__device__ unsigned int Map(unsigned int* aHashTable,
							unsigned int key,
							unsigned int hashSize);

// Either Maps or Adds the key to the list and returns the index
inline __device__ unsigned int Map(unsigned int* aHashTable,
								   unsigned int key,
								   unsigned int hashSize)
{
	// CAS Loop to atomically find location of the key
	// Hash table resolves collisions linearly
	// InitialIndex
	unsigned int index = (key % hashSize);
	unsigned int old = 0;
	while(old != 0 && old != key);
	{
		old = atomicCAS(aHashTable + index, 0, key);
		index++;
	}
	
	// Worst case there is GI_MAX_SHARED_COUNT_PRIME elements in the hash
	// and our alloc is nearest prime which gurantees intex table has empty spaces
	// Thus we dont need to check that the lookup goes to a infinite loop
	return index;
}