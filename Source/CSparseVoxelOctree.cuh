/**

Sparse voxel octree implementation
Designed for fast reconstruction from its bottom 

*/

#ifndef __CSPARSEVOXELOCTREE_H__
#define __CSPARSEVOXELOCTREE_H__

#include <cuda_runtime.h>
#include <cuda.h>
#include "CVoxel.cuh"

// first int has
// first 24 bit is children index
// last 8 bit used to determine which children is avail
// --
// last 4 byte is used for color
typedef unsigned int CSVONode;
typedef unsigned int CSVOColor;

struct CSVOConstants
{
	unsigned int denseDim;
	unsigned int denseDepth;
	unsigned int totalDepth;
	unsigned int numCascades;
};

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

inline __device__ unsigned int CalculateChildIndex(const unsigned char childrenBits,
												   const unsigned char childBit)
{
	unsigned int childrenCount = __popc(childrenBits);
	unsigned int bit = childrenBits, totalBitCount = 0;
	for(unsigned int i = 0; i < childrenCount; i++)
	{
		totalBitCount += __ffs(bit);
		if((0x00000001 << totalBitCount) == childBit)
			return i;
		bit = bit >> (__ffs(bit) + 1);
		totalBitCount++;
	}
}

inline __device__ unsigned char CalculateLevelChildBit(const uint3& voxelPos,
													   const unsigned int levelDepth,
													   const unsigned int totalDepth)
{
	unsigned int bitSet = 0;
	bitSet |= ((voxelPos.z >> (totalDepth - levelDepth)) & 0x000000001) << 2;
	bitSet |= ((voxelPos.y >> (totalDepth - levelDepth)) & 0x000000001) << 1;
	bitSet |= ((voxelPos.x >> (totalDepth - levelDepth)) & 0x000000001) << 0;
	return (0x01 << static_cast<unsigned int>(bitSet));
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
		expandedVoxId.x = (~(expandedVoxId.x >> (8 + i)) << (8 + i + 1)) | expandedVoxId.x;
		expandedVoxId.y = (~(expandedVoxId.y >> (8 + i)) << (8 + i + 1)) | expandedVoxId.y;
		expandedVoxId.z = (~(expandedVoxId.z >> (8 + i)) << (8 + i + 1)) | expandedVoxId.z;
	}
	return expandedVoxId;
}

inline __device__ uint3 UnpackLevelVoxId(const unsigned int packedVoxel,
										 const unsigned int levelDepth,
										 const unsigned int cascadeNo,
										 const unsigned int numCascades)
{
	uint3 levelVoxelId = ExpandOnlyVoxPos(packedVoxel);
	if(cascadeNo > 0)
		return ExpandToSVODepth(levelVoxelId, cascadeNo, numCascades);
	return levelVoxelId;
}

// Returns the intersected voxelIndex if voxel is found
// Else it returns -1 (0xFFFFFFFF)
inline __device__ unsigned int ConeTrace() {return 0;}
#endif //__CSPARSEVOXELOCTREE_H__