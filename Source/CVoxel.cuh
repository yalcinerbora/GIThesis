/**

Voxel Sturcutres

*/

#ifndef __CVOXEL_H__
#define __CVOXEL_H__

#include "CVoxelTypes.h"

inline __device__ SegmentOccupation ExpandOnlyOccupation(const uint16_t packed)
{
	return static_cast<SegmentOccupation>((packed >> 11) & 0x0007);
}

inline __device__ void ExpandSegmentObj(CVoxelObjectType& type,
										SegmentOccupation& occupation,
										uint16_t& segOccupancy,
										const uint16_t packed)
{
	// MSB to LSB 2 bit object type 2 bit
	type = static_cast<CVoxelObjectType>((packed >> 14) & 0x0003);
	occupation = ExpandOnlyOccupation(packed);
	segOccupancy = packed & 0x07FF;
}

inline __device__ uint3 ExpandVoxPos(bool& isMip, const CVoxelPos packedVoxX)
{
	isMip = ((packedVoxX & 0xC0000000) >> 30) != 0;
	uint3 result;
	result.x = (packedVoxX & 0x000003FF);
	result.y = (packedVoxX & 0x000FFC00) >> 10;
	result.z = (packedVoxX & 0x3FF00000) >> 20;
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

inline __device__ void ExpandOccupancy(uint3& neigbourBits, float3& weights,
									   const CVoxelOccupancy packedOccup)
{
	weights.x = static_cast<float>((packedOccup & 0x000000FF) >> 0) / 255.0f;
	weights.y = static_cast<float>((packedOccup & 0x0000FF00) >> 8) / 255.0f;
	weights.z = static_cast<float>((packedOccup & 0x00FF0000) >> 16) / 255.0f;
	neigbourBits.x = (packedOccup & 0x01000000) >> 24;
	neigbourBits.y = (packedOccup & 0x02000000) >> 25;
	neigbourBits.z = (packedOccup & 0x04000000) >> 26;
}

inline __device__ ushort2 ExpandOnlyObjId(const unsigned int packVoxIdX)
{
	ushort2 result;
	result.x = (packVoxIdX & 0x0000FFFF);
	result.y = (packVoxIdX & 0x3FFF0000) >> 16;
	return result;
}

//-------------------------------------------------------------------------------------------//

inline __device__ CVoxelPos PackVoxPos(const uint3& voxPos, const bool isMip)
{
	unsigned int packed = 0;
	unsigned int uintMip = (isMip) ? 1 : 0;
	packed |= (uintMip & 0x00000003) << 30;
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

inline __device__ CVoxelOccupancy PackOccupancy(const uint3& neigbourBits, const float3 weights)
{
	unsigned int result;
	result = neigbourBits.z << 26;
	result |= neigbourBits.y << 25;
	result |= neigbourBits.x << 24;
	result |= static_cast<unsigned int>(weights.z * 255.0f) << 16;
	result |= static_cast<unsigned int>(weights.y * 255.0f) << 8;
	result |= static_cast<unsigned int>(weights.x * 255.0f) << 0;
	return result;
}

inline __device__ uint16_t PackSegmentObj(const CVoxelObjectType type,
										  const SegmentOccupation occupation,
										  const uint16_t segOccupancy)
{
	//
	uint16_t packed = 0;
	packed |= (static_cast<uint16_t>(type) & 0x0003) << 14;
	packed |= (static_cast<uint16_t>(occupation) & 0x0007) << 11;
	packed |= (segOccupancy & 0x07FF) << 0;
	return packed;
}

#endif //__CVOXEL_H__