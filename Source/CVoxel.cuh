/**

Voxel Sturcutres

*/

#ifndef __CVOXEL_H__
#define __CVOXEL_H__

#include "CVoxelTypes.h"
#include "CVoxelPage.h"

inline __device__ SegmentOccupation ExpandOnlyOccupation(const uint16_t packed)
{
	return static_cast<SegmentOccupation>((packed >> 11) & 0x0007);
}

inline __device__ void ExpandSegmentPacked(CVoxelObjectType& type,
										   SegmentOccupation& occupation,
										   uint16_t& segOccupancy,
										   const uint16_t packed)
{
	// MSB to LSB 2 bit object type 2 bit
	type = static_cast<CVoxelObjectType>((packed >> 14) & 0x0003);
	occupation = ExpandOnlyOccupation(packed);
	segOccupancy = packed & 0x07FF;
}

inline __device__ uint16_t PackSegmentPacked(const CVoxelObjectType type,
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

inline __device__ uint3 ExpandOnlyVoxPos(const unsigned int packedVoxX)
{
	uint3 result;
	result.x = (packedVoxX & 0x000003FF);
	result.y = (packedVoxX & 0x000FFC00) >> 10;
	result.z = (packedVoxX & 0x3FF00000) >> 20;
	return result;
}

inline __device__ float4 ExpandOnlyNormal(const unsigned int packedVoxY)
{
	float4 result;
	result.x = static_cast<float>(static_cast<char>((packedVoxY >>  0) & 0xFF)) / 0x7F;
	result.y = static_cast<float>(static_cast<char>((packedVoxY >>  8) & 0xFF)) / 0x7F;
	result.z = static_cast<float>(static_cast<char>((packedVoxY >> 16) & 0xFF)) / 0x7F;
	result.w = static_cast<float>((packedVoxY & 0xFF000000) >> 24) / 255.0f;
	return result;
}

inline __device__ void ExpandNormalPos(uint3& voxPos,
									   float4& normal,
									   bool& isMip,
									   const CVoxelNormPos& packedVoxNormalPos)
{
	unsigned int voxPosX = packedVoxNormalPos.x;
	voxPos = ExpandOnlyVoxPos(voxPosX);
	isMip = ((voxPosX & 0xC0000000) >> 30) != 0;
	normal = ExpandOnlyNormal(packedVoxNormalPos.y);
}

inline __device__ ushort2 ExpandOnlyObjId(const unsigned int packVoxIdX)
{
	ushort2 result;
	result.x = (packVoxIdX & 0x0000FFFF);
	result.y = (packVoxIdX & 0x3FFF0000) >> 16;
	return result;
}

inline __device__ void ExpandVoxelIds(unsigned int& voxId,
									  ushort2& objectId,
									  CVoxelObjectType& objType,
									  const CVoxelIds& packedVoxIds)
{
	unsigned int voxIdX = packedVoxIds.x;
	objectId = ExpandOnlyObjId(voxIdX);
	objType = static_cast<CVoxelObjectType>((voxIdX & 0xC0000000) >> 30);
	voxId = packedVoxIds.y;
}

inline __device__ void PackVoxelIds(CVoxelIds& packedVoxId,
									const ushort2& objId,
									const CVoxelObjectType& objType,
									const unsigned int voxRenderPtr)
{
	// 3rd word holds object id (14/16 bit each)
	// 1st is batch id second is object id on that batch
	unsigned int value = 0;
	value |= static_cast<unsigned int>(objType) << 30;
	value |= static_cast<unsigned int>(objId.y) << 16;
	value |= static_cast<unsigned int>(objId.x);
	packedVoxId.x = value;

	// Last component is voxel render index on that batch precalculated voxel batch
	packedVoxId.y = voxRenderPtr;
}

inline __device__ unsigned int PackOnlyVoxPos(const uint3& voxPos,
											  const bool isMip)
{
	unsigned int packed = 0;
	unsigned int uintMip = (isMip) ? 1 : 0;
	packed |= (uintMip & 0x00000003) << 30;
	packed |= (voxPos.z & 0x000003FF) << 20;
	packed |= (voxPos.y & 0x000003FF) << 10;
	packed |= (voxPos.x & 0x000003FF);
	return packed;
}

inline __device__ unsigned int PackOnlyVoxNorm(const float4& normal)
{
	unsigned int value = 0;
	value |= static_cast<unsigned int>(normal.w * 255.0f) << 24;
	value |= (static_cast<int>(normal.z * 0x7F) & 0xFF) << 16;
	value |= (static_cast<int>(normal.y * 0x7F) & 0xFF) << 8;
	value |= (static_cast<int>(normal.x * 0x7F) & 0xFF) << 0;
	return value;
}

inline __device__ void PackVoxelNormPos(CVoxelNormPos& packedVoxNormPos,
										const uint3& voxPos,
										const float4& normal,
										const bool isMip)
{
	// First word holds span ratio and voxel position (relative to AABB or Grid)
	packedVoxNormPos.x = PackOnlyVoxPos(voxPos, isMip);

	// Second word holds normal 
	packedVoxNormPos.y = PackOnlyVoxNorm(normal);
}
#endif //__CVOXEL_H__