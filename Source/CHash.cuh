/*

Quickly Implemented Hashtable

operates on storing transform matrices into shared memory
in page system

*/

#include <cuda_runtime.h>

// Using ahred memory on 16Kb config
#define GI_CUDA_SHARED_MEM_SIZE (48 * 1024)

// Max Worst case scenario for shared memory
#define GI_MAX_SHARED_COUNT 64
#define GI_MAX_SHARED_COUNT_PRIME 67

// Mapping and "Allocating" the objectId
// Map functions manages memory conflicts internally thus can be called
// from different threads at the same time
__device__ unsigned int Map(const ushort2& objectId, unsigned int* sHashIndex);
