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
__device__ void HashMap(// Hash table
						CVoxelPos* sMetaNodeTable,
						uint32_t* sMetaNodeBitTable,
						// Linear storage of occupied locations of hash table
						uint32_t* sOccupiedHashSpots,
						uint32_t& sHashSpotAllocator,
						// Key value
						CVoxelPos key,
						uint8_t keyBits,
						// Hash table limits
						int32_t tableSize);

inline __device__ void HashMap(// Hash table
							   CVoxelPos* sMetaNodeTable,
							   uint32_t* sMetaNodeBitTable,
							   // Linear storage of occupied locations of hash table
							   uint32_t* sOccupiedHashSpots,
							   uint32_t& sHashSpotAllocator,
							   // Key value
							   const CVoxelPos key,
							   const uint8_t keyBits,
							   // Hash table limits
							   const int32_t tableSize)
{
	// 0xFFFFFFFF is used to indicate empty space
	assert(key != 0xFFFFFFFF);

	// CAS Loop to atomically find location of the key
	// Hash table resolves collisions linearly	
	uint32_t index = HashFunction(key, tableSize);
	for(int32_t i = 0; i < tableSize; i++)
	{
		CVoxelPos old = atomicCAS(sMetaNodeTable + index, 0xFFFFFFFF, key);
		if(old == 0xFFFFFFFF || old == key)
		{
			// Apply your meta bits (how much expansion is needed from this meta node)
			atomicAnd(sMetaNodeBitTable + index, keyBits);
			if(old == 0xFFFFFFFF)
			{
				uint32_t spot = atomicAdd(&sHashSpotAllocator, 1);
				assert(spot < static_cast<uint32_t>(tableSize));
				sOccupiedHashSpots[spot] = index;
			}
			return;
		}
		// Quadratic Probing
		//int32_t offset = (i + 1) * (i + 1) * (((i + 1) % 2 == 0) ? -1 : 1);
		//index = (HashFunction(key, tableSize) + offset) % tableSize;
		// Linear Probing
		index = (HashFunction(key, tableSize) + i + 1) % tableSize;
	}
	// We couldnt be able to add node,
	// this should never happen hash table is always sufficiently large.
	assert(false);
	printf("Failed Hash!");
}

inline __device__ uint32_t HashFunction(const CVoxelPos& key,
										uint32_t tableSize)
{
	// TODO:: Change to morton based z order curve
	return key % tableSize;
}

inline __device__ void HashTableReset(uint32_t& sHashSpotAllocator,
									  CVoxelPos* sMetaNodes,
									  uint32_t* sMetaNodeBitmap,
									  const uint32_t HashSize)
{
	if(threadIdx.x == 0) sHashSpotAllocator = 0;
	sMetaNodes[threadIdx.x] = 0xFFFFFFFF;
	sMetaNodeBitmap[threadIdx.x] = 0xFFFFFFFF;
	if(threadIdx.x < (HashSize % blockDim.x))
	{
		sMetaNodes[threadIdx.x + blockDim.x] = 0xFFFFFFFF;
		sMetaNodeBitmap[threadIdx.x + blockDim.x] = 0xFFFFFFFF;
	}
}