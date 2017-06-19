#pragma once
/**

Sparse voxel octree implementation
Designed for fast reconstruction from its bottom 

*/

#include <cuda.h>
#include <cuda_fp16.h>
#include "CSVOTypes.h"
#include "CVoxelFunctions.cuh"
#include <cassert>
#include <cstdio>

// Small Tidy Functions
inline __device__ unsigned int DenseIndex(const uint3& voxelPos, const unsigned int levelSize)
{
	return  voxelPos.z * levelSize * levelSize +
			voxelPos.y * levelSize +
			voxelPos.x;
}

// Illumination Storage
inline __device__ float4 UnpackSVOIrradiance(const CSVOIrradiance& irrad)
{
	float4 irradiance;
	irradiance.x = static_cast<float>((irrad & 0x000000FF) >> 0) / 255.0f;
	irradiance.y = static_cast<float>((irrad & 0x0000FF00) >> 8) / 255.0f;
	irradiance.z = static_cast<float>((irrad & 0x00FF0000) >> 16) / 255.0f;
	irradiance.w = static_cast<float>((irrad & 0xFF000000) >> 24) / 255.0f;
	return irradiance;
}

inline __device__ float4 UnpackSVOOccupancy(const CSVOWeight& weight)
{
	return UnpackSVOIrradiance(weight);
}

inline __device__ float4 UnpackSVONormal(const CSVONormal& normal)
{
	float4 result;
	result.x = static_cast<float>(static_cast<char>((normal >> 0) & 0xFF)) / 0x7F;
	result.y = static_cast<float>(static_cast<char>((normal >> 8) & 0xFF)) / 0x7F;
	result.z = static_cast<float>(static_cast<char>((normal >> 16) & 0xFF)) / 0x7F;
	result.w = static_cast<float>(static_cast<unsigned char>((normal >> 24) & 0xFF));
	return result;
}

inline __device__ float4 UnpackSVOLightDir(const CSVOLightDir& lightDir)
{
	return UnpackSVONormal(lightDir);
}

// Illumination Unpack
inline __device__ CSVOIrradiance PackSVOIrradiance(const float4& irradiance)
{
	CSVOIrradiance packed;
	packed = static_cast<unsigned int>(irradiance.w * 255.0f) << 24;
	packed |= static_cast<unsigned int>(irradiance.z * 255.0f) << 16;
	packed |= static_cast<unsigned int>(irradiance.y * 255.0f) << 8;
	packed |= static_cast<unsigned int>(irradiance.x * 255.0f) << 0;
	return packed;
}

inline __device__ CSVOWeight PackSVOOccupancy(const float4& weight)
{
	return PackSVOIrradiance(weight);
}

inline __device__ CSVONormal PackSVONormal(const float4& normal)
{
	unsigned int value = 0;
	value |= (static_cast<unsigned int>(normal.w) & 0xFF) << 24;
	value |= (static_cast<int>(normal.z * 0x7F) & 0xFF) << 16;
	value |= (static_cast<int>(normal.y * 0x7F) & 0xFF) << 8;
	value |= (static_cast<int>(normal.x * 0x7F) & 0xFF) << 0;
	return value;
}

inline __device__ CSVOLightDir PackSVOLightDir(const float4& lightDir)
{
	return PackSVONormal(lightDir);
}

// 64-bit to 32-bit conversions
inline __device__ uint64_t PackWords(const uint32_t& upper,
									 const uint32_t& lower)
{
	uint64_t mat = 0;
	mat |= static_cast<uint64_t>(upper) << 32;
	mat |= static_cast<uint64_t>(lower);
	return mat;
}

inline __device__ unsigned int UnpackLowerWord(const uint64_t& portion)
{
	return static_cast<unsigned int>(portion & 0x00000000FFFFFFFF);
}

inline __device__ unsigned int UnpackUpperWord(const uint64_t& portion)
{
	return static_cast<unsigned int>((portion & 0xFFFFFFFF00000000) >> 32);
}

inline __device__ uint2 UnpackWords(const uint64_t& portion)
{
	return
	{
		UnpackLowerWord(portion),
		UnpackUpperWord(portion)
	};
}

// Node Manipulation
inline __device__ unsigned int CalculateLevelChildId(const uint3& voxelPos,
													 const unsigned int levelDepth,
													 const unsigned int maxSVODepth)
{
	unsigned int bitSet = 0;
	bitSet |= ((voxelPos.z >> (maxSVODepth - levelDepth)) & 0x000000001) << 2;
	bitSet |= ((voxelPos.y >> (maxSVODepth - levelDepth)) & 0x000000001) << 1;
	bitSet |= ((voxelPos.x >> (maxSVODepth - levelDepth)) & 0x000000001) << 0;
	return bitSet;
}

//inline __device__ unsigned char CalculateLevelChildBit(const uint3& voxelPos,
//													   const unsigned int levelDepth,
//													   const unsigned int totalDepth)
//{
//	return 0x01 << CalculateLevelChildId(voxelPos, levelDepth, totalDepth);
//}

inline __device__ uint3 CalculateLevelVoxId(const uint3& voxelPos,
											const unsigned int levelDepth,
											const unsigned int maxSVODepth)
{
	uint3 levelVoxelId;
	levelVoxelId.x = voxelPos.x >> (maxSVODepth - levelDepth);
	levelVoxelId.y = voxelPos.y >> (maxSVODepth - levelDepth);
	levelVoxelId.z = voxelPos.z >> (maxSVODepth - levelDepth);
	return levelVoxelId;
}

inline __device__ uint3 ExpandToSVODepth(const uint4& voxelPos,
										 const unsigned int numCascades,
										 const unsigned int baseLevel)
{
	unsigned int cascadeNo = voxelPos.w;
	unsigned int invCascadeNo = (numCascades - cascadeNo - 1);

	uint3 expandedVoxId;
	expandedVoxId.x = voxelPos.x << cascadeNo;
	expandedVoxId.y = voxelPos.y << cascadeNo;
	expandedVoxId.z = voxelPos.z << cascadeNo;

	for(unsigned int i = 0; i < invCascadeNo; i++)
	{
		// Bit expansion of inner cascades
		// if MSB is 1 it becomes 10
		// if MSB is 0 it becomes 01
		unsigned int bitLoc = baseLevel + cascadeNo + i;
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

inline __device__ CSVONode PackNodeId(const uint3& localVoxelPos,
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
	return PackVoxPos(voxId, 0);
}

inline __device__ uint3 UnpackNodeId(const CSVONode nodePacked,
									 const unsigned int level,
									 const unsigned int numCascades,
									 const unsigned int totalLevel)
{
	uint4 nodeId = ExpandVoxPos(nodePacked);
	unsigned int packLevel = totalLevel - numCascades + 1;
	unsigned int cascadeNo = fmaxf(level, packLevel) - packLevel;

	assert(nodeId.x < (0x1u << min(level, packLevel)));
	assert(nodeId.y < (0x1u << min(level, packLevel)));
	assert(nodeId.z < (0x1u << min(level, packLevel)));
	nodeId.w = cascadeNo;

	uint3 result = ExpandToSVODepth(nodeId, 
									numCascades, 
									totalLevel);

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

//inline __device__ unsigned int CalculateChildIndex(const unsigned char childrenBits,
//												   const unsigned char childBit)
//{
//	assert((childrenBits & childBit) != 0);	
//	return __popc(childrenBits & (childBit - 1));
//}