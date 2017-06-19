#pragma once
/*

Quickly Implemented Hashtable

operates on storing transform matrices into shared memory
in page system

*/

#include <cassert>
#include <cuda_runtime.h>
#include "CSVOTypes.h"

// Nearest small prime of power of twos
__device__ static constexpr unsigned int nearestPrimeofTwo[] =
{
	1,		// 2
	3,		// 4
	7,		// 8
	13,		// 16
	31,		// 32
	63,		// 64
	131,	// 128
	251,	// 256
	509,	// 512
	1021,   // 1024
	2039,	// 2048
	4093,	// 4096	
	6143	// 6144	Special Case (Maximum occupancy 4-byte data for maxwell (TBB=512))
	// Rest will drop occupancy so omitted
	// Will add later
};

// Mapping and "Allocating" the objectId
// Map functions manages memory conflicts internally thus can be called
// from different threads at the same time
__device__ bool Map(unsigned int& index,
					CSVONode* sHashTable,
					CSVONode key,
					unsigned int tableSize);

// Either Maps or Adds the key to the list and returns the index
inline __device__ bool Map(unsigned int& index,
						   CSVONode* sHashTable,
						   CSVONode key,
						   unsigned int tableSize)
{
	// CAS Loop to atomically find location of the key
	// Hash table resolves collisions linearly
	index = key % tableSize;	
	for(int i = 0; i < tableSize; i++)
	{
		CSVONode old = atomicCAS(sHashTable + index, 0xFFFFFFFF, key);
		
		if(old == 0xFFFFFFFF) return true;
		else if(old == key) return false;

		// Linear Probing
		index = (index + 1) % tableSize;
	}
	// We couldnt be able to add node (this should never happen hash tables always sufficiently large)
	assert(false);
	return false;
}