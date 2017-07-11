#pragma once

/**

Voxel Sturcutres

*/

#include "CVoxelTypes.h"

inline __device__ CSegmentOccupation ExpandOnlyOccupation(const uint16_t packed)
{
	return static_cast<CSegmentOccupation>((packed >> 10) & 0x0003);
}

inline __device__ uint8_t  ExpandOnlyCascadeNo(const uint16_t packed)
{
	return static_cast<uint8_t>((packed >> 14) & 0x0003);
}

inline __device__ void ExpandSegmentInfo(uint8_t& cascadId,
										 CObjectType& objType,
										 CSegmentOccupation& occupation,
										 bool& firstOccurance,
										 const uint16_t packed)
{
	// MSB to LSB 
	// 2 bit cascadeId
	// 2 bit object type 
	// 2 bit segment occupation
	cascadId = ExpandOnlyCascadeNo(packed);
	objType = static_cast<CObjectType>((packed >> 12) & 0x0003);	
	occupation = ExpandOnlyOccupation(packed);
	firstOccurance = ((packed >> 9) & 0x0001) != 0;
}

inline __device__ int3 ExpandVoxPos(const CVoxelPos packedVoxPos)
{
	int3 result;	
	result.x = (packedVoxPos & 0x000003FF) >>  0;
	result.y = (packedVoxPos & 0x000FFC00) >> 10;
	result.z = (packedVoxPos & 0x3FF00000) >> 20;
	//result.w = (packedVoxPos & 0xC0000000) >> 30;
	return result;
}

inline __device__ float3 ExpandVoxNormal(const CVoxelNorm packedVoxY)
{
	float3 result;
	result.x = static_cast<float>(static_cast<char>((packedVoxY >>  0) & 0xFF)) / 0x7F;
	result.y = static_cast<float>(static_cast<char>((packedVoxY >>  8) & 0xFF)) / 0x7F;
	result.z = static_cast<float>(static_cast<char>((packedVoxY >> 16) & 0xFF)) / 0x7F;
	return result;
}

inline __device__ float3 ExpandOccupancy(const CVoxelOccupancy packedOccup)
{
	float3 weights;
	weights.x = static_cast<float>((packedOccup & 0x000000FF) >> 0) / 255.0f;
	weights.y = static_cast<float>((packedOccup & 0x0000FF00) >> 8) / 255.0f;
	weights.z = static_cast<float>((packedOccup & 0x00FF0000) >> 16) / 255.0f;
	return weights;
	//neigbourBits.x = (packedOccup & 0x01000000) >> 24;
	//neigbourBits.y = (packedOccup & 0x02000000) >> 25;
	//neigbourBits.z = (packedOccup & 0x04000000) >> 26;
}

inline __device__ ushort2 ExpandOnlyObjId(const unsigned int packVoxIdX)
{
	ushort2 result;
	result.x = (packVoxIdX & 0x0000FFFF);
	result.y = (packVoxIdX & 0x3FFF0000) >> 16;
	return result;
}

//-------------------------------------------------------------------------------------------//

inline __device__ CVoxelPos PackVoxPos(const int3& voxPos/*, const unsigned int cascadeNo*/)
{
	unsigned int packed = 0;
	//packed |= (cascadeNo & 0x00000003) << 30;
	packed |= (voxPos.z & 0x000003FF) << 20;
	packed |= (voxPos.y & 0x000003FF) << 10;
	packed |= (voxPos.x & 0x000003FF);
	return packed;
}

inline __device__ CVoxelNorm PackVoxNormal(const float3& normal)
{
	unsigned int value = 0;
	value |= (static_cast<int>(normal.z * 0x7F) & 0xFF) << 16;
	value |= (static_cast<int>(normal.y * 0x7F) & 0xFF) << 8;
	value |= (static_cast<int>(normal.x * 0x7F) & 0xFF) << 0;
	return value;
}

inline __device__ CVoxelOccupancy PackOccupancy(const float3 weights)
{
	unsigned int result;
	result = static_cast<unsigned int>(weights.z * 255.0f) << 16;
	result |= static_cast<unsigned int>(weights.y * 255.0f) << 8;
	result |= static_cast<unsigned int>(weights.x * 255.0f) << 0;
	return result;
}

inline __device__ uint16_t PackSegmentInfo(const uint8_t cascadeId,
										   const CObjectType type,
										   const CSegmentOccupation occupation,
										   const bool firstOccurance)
{
	// MSB to LSB 
	// 2 bit cascadeId
	// 2 bit object type 
	// 2 bit segment occupation
	uint16_t packed = 0;
	packed |= (static_cast<uint16_t>(cascadeId) & 0x0003) << 14;
	packed |= (static_cast<uint16_t>(type) & 0x0003) << 12;
	packed |= (static_cast<uint16_t>(occupation) & 0x0003) << 10;
	packed |= (static_cast<uint16_t>(firstOccurance) & 0x0001) << 9;
	return packed;
}