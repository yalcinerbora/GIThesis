/**

Atomic Allocation used by voxel include and svo reconstruct
its basically a CAS looped integer value that uses an array as a stack

*/

#ifndef __CATOMICALLOC_H__
#define __CATOMICALLOC_H__

#include <cuda.h>

inline __device__ unsigned int AtomicAlloc(unsigned int* gStackSize)
{
	unsigned int assumed, old = *gStackSize;
	do
	{
		assumed = old;
		unsigned int result = (assumed == 0) ? 0 : (assumed - 1);
		old = atomicCAS(gStackSize, assumed, result);
	}
	while(assumed != old);
	return old;
}

inline __device__ unsigned int AtomicDealloc(unsigned int* gStackSize,
											 const unsigned int maxValue)
{
	unsigned int assumed, old = *gStackSize;
	do
	{
		assumed = old;
		unsigned int result = (assumed == GI_SEGMENT_PER_PAGE) ? maxValue : (assumed + 1);
		old = atomicCAS(gStackSize, assumed, result);
	}
	while(assumed != old);
	return old;
}
#endif //__CATOMICALLOC_H__