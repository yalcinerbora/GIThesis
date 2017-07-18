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
inline __device__ unsigned int DenseIndex(const int3& voxelPos, const unsigned int levelSize)
{
	return  voxelPos.z * levelSize * levelSize +
			voxelPos.y * levelSize +
			voxelPos.x;
}

inline __device__ unsigned int DenseIndex(const short3& voxelPos, const unsigned int levelSize)
{
	return  voxelPos.z * levelSize * levelSize +
			voxelPos.y * levelSize +
			voxelPos.x;
}

inline __device__ void Swap(char3& data, int8_t i0, int8_t i1)
{
	assert(0 <= i0 && i0 < 3);
	assert(0 <= i1 && i1 < 3);
	char temp = reinterpret_cast<char*>(&data)[i0];
	reinterpret_cast<char*>(&data)[i0] = reinterpret_cast<char*>(&data)[i1];
	reinterpret_cast<char*>(&data)[i1] = temp;
}

// Illumination Storage
inline __device__ float4 UnpackSVOIrradiance(const CSVOIrradiance& irrad)
{
	float4 irradiance;
	irradiance.x = static_cast<float>((irrad & 0x000000FF) >>  0) / 255.0f;
	irradiance.y = static_cast<float>((irrad & 0x0000FF00) >>  8) / 255.0f;
	irradiance.z = static_cast<float>((irrad & 0x00FF0000) >> 16) / 255.0f;
	irradiance.w = static_cast<float>((irrad & 0xFF000000) >> 24) / 255.0f;
	return irradiance;
}

inline __device__ float4 UnpackSVOOccupancy(const CSVOWeight& weight)
{
	return UnpackSVOIrradiance(weight);
}

inline __device__ float4 UnpackLightDirLeaf(const uint32_t& packed)
{
	float4 lightDir;
	lightDir.x = static_cast<float>(static_cast<char>((packed >>  0) & 0xFF)) / 0x7F;
	lightDir.y = static_cast<float>(static_cast<char>((packed >>  8) & 0xFF)) / 0x7F;
	lightDir.z = static_cast<float>(static_cast<char>((packed >> 16) & 0xFF)) / 0x7F;
	lightDir.w = static_cast<float>(static_cast<char>((packed >> 24) & 0xFF)) / 0x7F;
	return lightDir;
}

inline __device__ float3 UnpackSVOUpperLeaf(const uint32_t& packed)
{
	// Use the sign bit of counter as sign bit of normal.z
	float3 result;
	result.x = static_cast<float>(static_cast<char>((packed >> 0) & 0xFF)) / 0x7F;
	result.y = static_cast<float>(static_cast<char>((packed >> 8) & 0xFF)) / 0x7F;
	result.z = __half2float(__ushort_as_half(static_cast<uint16_t>((packed >> 16) & 0xFFFF)));
	return result;
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
	CSVOIrradiance packed = 0;
	packed |= static_cast<unsigned int>(irradiance.w * 255.0f) << 24;
	packed |= static_cast<unsigned int>(irradiance.z * 255.0f) << 16;
	packed |= static_cast<unsigned int>(irradiance.y * 255.0f) << 8;
	packed |= static_cast<unsigned int>(irradiance.x * 255.0f) << 0;
	return packed;
}

inline __device__ CSVOWeight PackSVOOccupancy(const float4& weight)
{
	return PackSVOIrradiance(weight);
}

inline __device__ uint32_t PackLightDirLeaf(const float4& lightDir)
{
	unsigned int packed = 0;
	packed |= (static_cast<int>(lightDir.w * 0x7F) & 0xFF) << 24;
	packed |= (static_cast<int>(lightDir.z * 0x7F) & 0xFF) << 16;
	packed |= (static_cast<int>(lightDir.y * 0x7F) & 0xFF) << 8;
	packed |= (static_cast<int>(lightDir.x * 0x7F) & 0xFF) << 0;
	return packed;
}

inline __device__ uint32_t PackSVOUpperLeaf(const float3& data)
{		
	uint32_t packed = 0;
	packed |= static_cast<uint32_t>(__half_as_ushort(__float2half(data.z))) << 16;
	packed |= (static_cast<int>(data.y * 0x7F) & 0x000000FF) << 8;
	packed |= (static_cast<int>(data.x * 0x7F) & 0x000000FF) << 0;
	return packed;
}

inline __device__ CSVONormal PackSVONormal(const float4& normal)
{
	unsigned int packed = 0;
	packed |= (static_cast<unsigned int>(normal.w) & 0xFF) << 24;
	packed |= (static_cast<int>(normal.z * 0x7F) & 0xFF) << 16;
	packed |= (static_cast<int>(normal.y * 0x7F) & 0xFF) << 8;
	packed |= (static_cast<int>(normal.x * 0x7F) & 0xFF) << 0;
	return packed;
}

inline __device__ CSVOLightDir PackSVOLightDir(const float4& lightDir)
{
	return PackSVONormal(lightDir);
}

// 64-bit to 32-bit conversions
inline __device__ uint64_t PackWords(const uint32_t& upper,
									 const uint32_t& lower)
{
	uint64_t illum = 0;
	illum |= static_cast<uint64_t>(upper) << 32;
	illum |= static_cast<uint64_t>(lower);
	return illum;
}

inline __device__ uint32_t UnpackLowerWord(const uint64_t& portion)
{
	return static_cast<unsigned int>(portion & 0x00000000FFFFFFFF);
}

inline __device__ uint32_t UnpackUpperWord(const uint64_t& portion)
{
	return static_cast<unsigned int>((portion & 0xFFFFFFFF00000000) >> 32);
}

inline __device__ uint2 UnpackWords(const uint64_t& portion)
{
	uint2 result;
	result.x = UnpackLowerWord(portion);
	result.y = UnpackUpperWord(portion);
	return result;
}

// Node Manipulation
inline __device__ unsigned int CalculateLevelChildId(const int3& voxelPos,
													 const unsigned int parentLevel,
													 const unsigned int currentLevel)
{
	unsigned int bitSet = 0;
	bitSet |= ((voxelPos.z >> (currentLevel - parentLevel)) & 0x000000001) << 2;
	bitSet |= ((voxelPos.y >> (currentLevel - parentLevel)) & 0x000000001) << 1;
	bitSet |= ((voxelPos.x >> (currentLevel - parentLevel)) & 0x000000001) << 0;
	return bitSet;
}

//inline __device__ unsigned char CalculateLevelChildBit(const uint3& voxelPos,
//													   const unsigned int levelDepth,
//													   const unsigned int totalDepth)
//{
//	return 0x01 << CalculateLevelChildId(voxelPos, levelDepth, totalDepth);
//}

inline __device__ int3 CalculateParentVoxId(const int3& voxelPos,
											 const unsigned int parentLevel,
											 const unsigned int currentLevel)
{
	assert(currentLevel >= parentLevel);
	int3 levelVoxelId;
	levelVoxelId.x = voxelPos.x >> (currentLevel - parentLevel);
	levelVoxelId.y = voxelPos.y >> (currentLevel - parentLevel);
	levelVoxelId.z = voxelPos.z >> (currentLevel - parentLevel);
	return levelVoxelId;
}

inline __device__ int3 ExpandToSVODepth(const int3& voxelPos,
										 const unsigned int cascadeId,
										 const unsigned int numCascades,
										 const unsigned int baseLevel)
{
	int cascadeNo = static_cast<int>(cascadeId);
	int invCascadeNo = static_cast<int>(numCascades) - cascadeNo - 1;

	int3 expandedVoxId;
	expandedVoxId.x = voxelPos.x;
	expandedVoxId.y = voxelPos.y;
	expandedVoxId.z = voxelPos.z;
	
	unsigned int bitLoc = baseLevel;
	unsigned int rightBitMask = (0x1 << (bitLoc - 1)) - 1;
	unsigned int expansionBits = 0x1 << invCascadeNo;
	unsigned int componentBit;
	
	if(invCascadeNo > 0)
	{
		// X
		componentBit = (expandedVoxId.x >> (bitLoc - 1)) & 0x1;
		componentBit = (componentBit == 0) ? (expansionBits - 1) : expansionBits;
		expandedVoxId.x &= rightBitMask;
		expandedVoxId.x |= componentBit << (bitLoc - 1);

		// Y
		componentBit = (expandedVoxId.y >> (bitLoc - 1)) & 0x1;
		componentBit = (componentBit == 0) ? (expansionBits - 1) : expansionBits;
		expandedVoxId.y &= rightBitMask;
		expandedVoxId.y |= componentBit << (bitLoc - 1);

		// Z
		componentBit = (expandedVoxId.z >> (bitLoc - 1)) & 0x1;
		componentBit = (componentBit == 0) ? (expansionBits - 1) : expansionBits;
		expandedVoxId.z &= rightBitMask;
		expandedVoxId.z |= componentBit << (bitLoc - 1);
	}
	return expandedVoxId;
}

inline __device__ CVoxelPos PackNodeId(const int3& localVoxelPos,
									   const unsigned int level,
									   const unsigned int numCascades,
									   const unsigned int baseLevel,
									   const unsigned int maxSVOLevel)
{
	unsigned int cascadeNo = maxSVOLevel - level;
	int3 result = localVoxelPos;

	// Pack it if it does not fit into baseLevel
	if(cascadeNo < numCascades - 1)
	{
		unsigned int bitLoc = baseLevel;
		unsigned int baseBitMask = (0x1 << (bitLoc - 1));
		unsigned int rightBitMask = baseBitMask - 1;
		
		// X
		result.x = (~localVoxelPos.x) & baseBitMask;
		result.x |= (localVoxelPos.x) & rightBitMask;

		// Y
		result.y = (~localVoxelPos.y) & baseBitMask;
		result.y |= (localVoxelPos.y) & rightBitMask;

		// Z
		result.z = (~localVoxelPos.z) & baseBitMask;
		result.z |= (localVoxelPos.z) & rightBitMask;
	}
	return PackVoxPos(result);
}

//inline __device__ unsigned int CalculateChildIndex(const unsigned char childrenBits,
//												   const unsigned char childBit)
//{
//	assert((childrenBits & childBit) != 0);	
//	return __popc(childrenBits & (childBit - 1));
//}