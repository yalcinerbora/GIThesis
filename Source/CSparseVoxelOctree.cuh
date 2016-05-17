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
#include <cstdio>

//inline __device__ void UnpackNode(unsigned int& childrenIndex,
//								  unsigned char& childrenBit,
//								  const CSVONode& node)
//{
//	childrenIndex = node & 0x00FFFFFF;
//	childrenBit = node >> 24;
//}
//
//inline __device__ CSVONode PackNode(const unsigned int& childrenIndex,
//									const unsigned char& childrenBit)
//{
//	CSVONode node;
//	node = childrenBit << 24;
//	node |= childrenIndex;
//	return node;
//}

inline __device__ float4 UnpackSVOColor(const CSVOColor& node)
{
	float4 color;
	color.x = static_cast<float>((node & 0x000000FF) >> 0) / 255.0f;
	color.y = static_cast<float>((node & 0x0000FF00) >> 8) / 255.0f;
	color.z = static_cast<float>((node & 0x00FF0000) >> 16) / 255.0f;
	color.w = static_cast<float>((node & 0xFF000000) >> 24) / 255.0f;
	return color;
}

inline __device__ float4 UnpackNormCount(const CVoxelNorm& normal)
{
	float4 result;
	result.x = static_cast<float>(static_cast<char>((normal >> 0) & 0xFF)) / 0x7F;
	result.y = static_cast<float>(static_cast<char>((normal >> 8) & 0xFF)) / 0x7F;
	result.z = static_cast<float>(static_cast<char>((normal >> 16) & 0xFF)) / 0x7F;
	result.w = static_cast<float>((normal >> 24) & 0xFF);
	return result;
}

inline __device__ CSVOColor PackSVOColor(const float4& color)
{
	CSVOColor colorPacked;
	colorPacked = static_cast<unsigned int>(color.w * 255.0f) << 24;
	colorPacked |= static_cast<unsigned int>(color.z * 255.0f) << 16;
	colorPacked |= static_cast<unsigned int>(color.y * 255.0f) << 8;
	colorPacked |= static_cast<unsigned int>(color.x * 255.0f) << 0;
	return colorPacked;
}

inline __device__ CVoxelNorm PackNormCount(const float4 normal)
{
	// (x,y components packed NORM int with 16/15 bit repectively, MSB is sign of z
	unsigned int value = 0;
	value |= static_cast<unsigned int>(normal.w) << 24;
	value |= (static_cast<int>(normal.z * 0x7F) & 0xFF) << 16;
	value |= (static_cast<int>(normal.y * 0x7F) & 0xFF) << 8;
	value |= (static_cast<int>(normal.x * 0x7F) & 0xFF) << 0;
	return value;
}

inline __device__ CSVOMaterial PackSVOMaterial(const CSVOColor& color,
											   const CVoxelNorm& normal)
{
	CSVOMaterial mat = 0;
	mat |= static_cast<CSVOMaterial>(normal) << 32;
	mat |= color;
	return mat;
}

inline __device__ void UnpackSVOMaterial(CSVOColor& color,
										 CVoxelNorm& normal,
										 const CSVOMaterial& mat)
{
	color = static_cast<CVoxelNorm>(mat & 0xFFFFFFFF);
	normal = static_cast<CVoxelNorm>(mat >> 32);
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
	return 0x01 << CalculateLevelChildId(voxelPos, levelDepth, totalDepth);
}

inline __device__ uint3 CalculateLevelVoxId(const uint3& voxelPos,
											const unsigned int levelDepth,
											const unsigned int totalDepth)
{
	uint3 levelVoxelId;
	levelVoxelId.x = voxelPos.x >> (totalDepth - levelDepth);
	levelVoxelId.y = voxelPos.y >> (totalDepth - levelDepth);
	levelVoxelId.z = voxelPos.z >> (totalDepth - levelDepth);
	return levelVoxelId;
}

inline __device__ uint3 ExpandToSVODepth(const uint3& localVoxelPos,
										 const unsigned int cascadeNo,
										 const unsigned int numCascades,
										 const unsigned int totalLevel)
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
		unsigned int bitLoc = totalLevel - cascadeNo + i;
		unsigned int rightBitMask = (0x01 << (bitLoc - 1)) - 1;
		unsigned int componentBit;
		unsigned int component;

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

inline __device__ unsigned int PackNodeId(const uint3& localVoxelPos,
										  const unsigned int level,
										  const unsigned int numCascades,
										  const unsigned int totalLevel)
{
	// Pack Level
	unsigned int packLevel = totalLevel - numCascades + 1;
	unsigned int packMaskLow = (0x1 << packLevel - 1) - 1;

	// Shift to Level
	uint3 voxId = CalculateLevelVoxId(localVoxelPos, level, totalLevel);

	// We need to pack stuff in order to open it properly
	unsigned int lastBit;
	if(level > packLevel)
	{
		// Flip the "packLevel"th bit
		// Zero out the left of it
		lastBit = voxId.x >> (packLevel - 1) & 0x1u;
		lastBit = 1 - lastBit;
		voxId.x = (lastBit << (packLevel - 1)) | (voxId.x & packMaskLow);
		
		lastBit = voxId.y >> (packLevel - 1) & 0x1u;
		lastBit = 1 - lastBit;
		voxId.y = (lastBit << (packLevel - 1)) | (voxId.y & packMaskLow);

		lastBit = voxId.z >> (packLevel - 1) & 0x1u;
		lastBit = 1 - lastBit;
		voxId.z = (lastBit << (packLevel - 1)) | (voxId.z & packMaskLow);
	}
	
	assert(voxId.x < (0x1u << min(level, packLevel)));
	assert(voxId.y < (0x1u << min(level, packLevel)));
	assert(voxId.z < (0x1u << min(level, packLevel)));
	return PackOnlyVoxPos(voxId, false);
}

inline __device__ uint3 UnpackNodeId(const unsigned int nodePacked,
									 const unsigned int level,
									 const unsigned int numCascades,
									 const unsigned int totalLevel)
{
	uint3 nodeId = ExpandOnlyVoxPos(nodePacked);
	unsigned int packLevel = totalLevel - numCascades + 1;
	unsigned int cascadeNo = fmaxf(level, packLevel) - packLevel;

	assert(nodeId.x < (0x1u << min(level, packLevel)));
	assert(nodeId.y < (0x1u << min(level, packLevel)));
	assert(nodeId.z < (0x1u << min(level, packLevel)));

	uint3 result = ExpandToSVODepth(nodeId, cascadeNo, numCascades, totalLevel);

	result.x >>= (numCascades - cascadeNo - 1);
	result.y >>= (numCascades - cascadeNo - 1);
	result.z >>= (numCascades - cascadeNo - 1);

	if(result.x >= (0x1 << level) ||
	   result.y >= (0x1 << level) ||
	   result.z >= (0x1 << level))
	{
		printf("voxelNode : %d, %d, %d\n"
			   "node : %d\n"
			   "nodeId : %d, %d, %d\n"
			   "------------\n",
			   result.x, result.y, result.z,
			   nodePacked,
			   nodeId.x, nodeId.y, nodeId.z
			   );
		assert(false);
	}
	assert(result.x < (0x1 << level));
	assert(result.y < (0x1 << level));
	assert(result.z < (0x1 << level));
	return result;
}

inline __device__ unsigned int CalculateChildIndex(const unsigned char childrenBits,
												   const unsigned char childBit)
{
	assert((childrenBits & childBit) != 0);	
	return __popc(childrenBits & (childBit - 1));
}
#endif //__CSPARSEVOXELOCTREE_H__ 