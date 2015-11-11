/**

Sparse voxel octree implementation
Designed for fast reconstruction from its bottom 

*/

#ifndef __CSPARSEVOXELOCTREE_H__
#define __CSPARSEVOXELOCTREE_H__

#include <cuda.h>
#include "CVoxel.cuh"
#include "CSVOTypes.cuh"
#include <cassert>

inline __device__ void UnpackNode(unsigned int& childrenIndex,
								  unsigned char& childrenBit,
								  const CSVONode& node)
{
	childrenIndex = node & 0x00FFFFFF;
	childrenBit = node >> 24;
}

inline __device__ CSVONode PackNode(const unsigned int& childrenIndex,
									const unsigned char& childrenBit)
{
	CSVONode node;
	node = childrenBit << 24;
	node |= childrenIndex;
	return node;
}

inline __device__ float4 UnpackSVOColor(const CSVOColor& node)
{
	float4 color;
	color.x = static_cast<float>((node & 0x000000FF) >> 0) / 255.0f;
	color.y = static_cast<float>((node & 0x0000FF00) >> 8) / 255.0f;
	color.z = static_cast<float>((node & 0x00FF0000) >> 16) / 255.0f;
	color.w = static_cast<float>((node & 0xFF000000) >> 24);
	return color;
}

inline __device__ CSVOColor PackSVOColor(const float4& color)
{
	CSVOColor colorPacked;
	colorPacked = static_cast<unsigned int>(color.w) << 24;
	colorPacked |= static_cast<unsigned int>(color.z * 255.0f) << 16;
	colorPacked |= static_cast<unsigned int>(color.y * 255.0f) << 8;
	colorPacked |= static_cast<unsigned int>(color.x * 255.0f) << 0;
	return colorPacked;
}

inline __device__ unsigned int CalculateLevelChildId(const uint3& voxelPos,
													 const unsigned int levelDepth,
													 const unsigned int totalDepth)
{
	unsigned int bitSet = 0;
	bitSet |= ((voxelPos.z >> (totalDepth - levelDepth)) & 0x000000001) << 2;
	bitSet |= ((voxelPos.y >> (totalDepth - levelDepth)) & 0x000000001) << 1;
	bitSet |= ((voxelPos.x >> (totalDepth - levelDepth)) & 0x000000001) << 0;
	return bitSet;
}

inline __device__ unsigned char CalculateLevelChildBit(const uint3& voxelPos,
													   const unsigned int levelDepth,
													   const unsigned int totalDepth)
{
	unsigned int bitSet = 0;
	bitSet |= ((voxelPos.z >> (totalDepth - levelDepth)) & 0x000000001) << 2;
	bitSet |= ((voxelPos.y >> (totalDepth - levelDepth)) & 0x000000001) << 1;
	bitSet |= ((voxelPos.x >> (totalDepth - levelDepth)) & 0x000000001) << 0;
	return (0x01 << (bitSet));
}

inline __device__ uint3 CalculateLevelVoxId(const uint3& voxelPos,
											const unsigned int levelDepth,
											const unsigned int totalDepth)
{
	uint3 levelVoxelId;
	levelVoxelId.x = (voxelPos.x >> (totalDepth - levelDepth));
	levelVoxelId.y = (voxelPos.y >> (totalDepth - levelDepth));
	levelVoxelId.z = (voxelPos.z >> (totalDepth - levelDepth));
	return levelVoxelId;
}

inline __device__ uint3 ExpandToSVODepth(const uint3& localVoxelPos,
										 const unsigned int cascadeNo,
										 const unsigned int numCascades)
{
	uint3 expandedVoxId = localVoxelPos;
	expandedVoxId.x = expandedVoxId.x << (numCascades - cascadeNo - 1);
	expandedVoxId.y = expandedVoxId.y << (numCascades - cascadeNo - 1);
	expandedVoxId.z = expandedVoxId.z << (numCascades - cascadeNo - 1);

	for(unsigned int i = 0; i < cascadeNo; i++)
	{
		// Bit expansion of inner cascades
		// if MSB is 1 it becomes 10
		// if MSB is 0 it becomes 01
		volatile unsigned int bitLoc = 9 + numCascades - cascadeNo - 1;
		volatile unsigned int rightBitMask = (0x01 << (bitLoc - 1)) - 1;
		volatile unsigned int componentBit;
		volatile unsigned int component;

		componentBit = expandedVoxId.x >> (bitLoc - 1);
		component = (1 - componentBit) * 0x01 + componentBit * 0x02;
		expandedVoxId.x = (component << (bitLoc - 1)) | 
							(expandedVoxId.x & rightBitMask);

		componentBit = expandedVoxId.y >> (bitLoc - 1);
		component = (1 - componentBit) * 0x01 + componentBit * 0x02;
		expandedVoxId.y = (component << (bitLoc - 1)) | 
							(expandedVoxId.y & rightBitMask);

		componentBit = expandedVoxId.z >> (bitLoc - 1);
		component = (1 - componentBit) * 0x01 + componentBit * 0x02;
		expandedVoxId.z = (component << (bitLoc - 1)) | 
							(expandedVoxId.z & rightBitMask);
	}
	return expandedVoxId;
}

inline __device__ unsigned int CalculateChildIndex(const unsigned char childrenBits,
												   const unsigned char childBit)
{
	assert((childrenBits & childBit) != 0);	
	return __popc(childrenBits & (childBit - 1));
}

inline __device__ unsigned int PosToKey(const uint3& levelVoxPos,
										const unsigned int level)
{
	unsigned int packedId = 0x00000000;
	unsigned int bitMask = 0x00000001;
	for(unsigned int i = 0; i < umax(level, 9u); i++)
	{
		packedId |= (levelVoxPos.x & (bitMask << i)) << (i * 2 + 0);
		packedId |= (levelVoxPos.y & (bitMask << i)) << (i * 2 + 1);
		packedId |= (levelVoxPos.z & (bitMask << i)) << (i * 2 + 2);
	}
	return packedId;

	//// TODO: better hash that occupies more warps
	//return PackOnlyVoxPos(levelVoxPos, 0);
}

inline __device__ uint3 KeyToPos(const unsigned int packedVoxel,
								 const unsigned int level,
								 const unsigned int cascadeNo,
								 const unsigned int numCascades)
{
	uint3 levelVoxelId = { 0, 0, 0 };
	unsigned int bitMask = 0x00000001;
	for(unsigned int i = 0; i < umax(level, 9u); i++)
	{
		levelVoxelId.x |= (packedVoxel >> (i * 2 + 0)) & (bitMask << i);
		levelVoxelId.y |= (packedVoxel >> (i * 2 + 1)) & (bitMask << i);
		levelVoxelId.z |= (packedVoxel >> (i * 2 + 2)) & (bitMask << i);
	}
	if(level > 9)
		return ExpandToSVODepth(levelVoxelId, cascadeNo, numCascades);
	return levelVoxelId;
}

// Returns the intersected voxelIndex if voxel is found
// Else it returns -1 (0xFFFFFFFF)
inline __device__ unsigned int ConeTrace() {return 0;}
#endif //__CSPARSEVOXELOCTREE_H__