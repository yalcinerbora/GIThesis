#pragma once
/*

Quickly Implemented Hashtable

operates on storing transform matrices into shared memory
in page system

*/

#include <cassert>
#include <cuda_runtime.h>
#include "CSVOTypes.h"
#include "CVoxelTypes.h"

// Hash Function
__device__ uint32_t HashFunction(const CVoxelPos& key,
								 uint32_t tableSize);

// Resets Hash Table
__device__ void HashTableReset(uint32_t& sHashSpotAllocator,
							   CVoxelPos* sMetaNodes,
							   uint32_t* sMetaNodeBitmap,
							   const uint32_t HashSize);

// Mapping and "Allocating" the objectId
// Either Maps or Adds the key to the list and returns the index
// Modified negbouring parent bitmap
__device__ bool HashMap(uint32_t& index,
						// Hash table							   
						uint32_t* sHashTable,
						// Key value
						const CVoxelPos key,
						// Hash table limits
						const int32_t tableSize);

inline __device__ bool HashMap(uint32_t& index,
							   // Hash table							   
							   uint32_t* sHashTable,
							   // Key value
							   const CVoxelPos key,
							   // Hash table limits
							   const int32_t tableSize)
{
	// 0xFFFFFFFF is used to indicate empty space
	assert(key != 0xFFFFFFFF);

	// CAS Loop to atomically find location of the key
	// Hash table resolves collisions linearly	
	index = HashFunction(key, tableSize);
	for(int32_t i = 0; i < tableSize; i++)
	{
		CVoxelPos old = atomicCAS(sHashTable + index, 0xFFFFFFFF, key);
		if(old == 0xFFFFFFFF) return true;
		if(old == key) return false;

		// Quadratic Probing (with guaranteed entire table traversal)
		int32_t offset = (i + 1) * (i + 1) * (((i + 1) % 2 == 0) ? -1 : 1);
		index = (HashFunction(key, tableSize) + offset) % tableSize;
		// Linear Probing
		//index = (HashFunction(key, tableSize) + i + 1) % tableSize;
	}
	// We couldnt be able to add node,
	// this should never happen hash table is always sufficiently large.
	assert(false);
	printf("Failed Hash!");
	return false;
}

inline __device__ uint32_t HashFunction(const CVoxelPos& key,
										uint32_t tableSize)
{
	// TODO:: Change to morton based z order curve
	return key % tableSize;
}

inline __device__ void HashTableReset(uint32_t* sHashTable,
									  const uint32_t HashSize)
{
	uint32_t iterationCount = (HashSize + blockDim.x - 1) / blockDim.x;
	for(uint32_t i = 0; i < iterationCount; i++)
	{		
		uint32_t sharedMemId = i * blockDim.x + threadIdx.x;
		if(sharedMemId < HashSize)
		{
			sHashTable[sharedMemId] = 0xFFFFFFFF;
		}
	}
}