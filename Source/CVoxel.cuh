/**

Voxel Sturcutres

*/

#ifndef __CVOXEL_H__
#define __CVOXEL_H__

#include "CVoxelTypes.h"

//
inline __device__ uint3 ExpandOnlyVoxPos(const unsigned int packedVoxX)
{
	uint3 result;
	result.x = (packedVoxX & 0x000001FF);
	result.y = (packedVoxX & 0x0003FE00) >> 9;
	result.z = (packedVoxX & 0x07FC0000) >> 18;
	return result;
}

inline __device__ float3 ExpandOnlyNormal(const unsigned int packedVoxY)
{
	float3 result;
	result.x = ((static_cast<float>(packedVoxY & 0xFFFF) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((static_cast<float>((packedVoxY >> 16) & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrtf(fabsf(1.0f - (result.x * result.x + result.y * result.y)));
	result.z *= (packedVoxY >> 31 == 1) ? -1.0f : 1.0f;
	return result;
}

inline __device__ void ExpandNormalPos(uint3& voxPos,
									   float3& normal,
									   bool& isMip,
									   const CVoxelNormPos& packedVoxNormalPos)
{
	unsigned int voxPosX = packedVoxNormalPos.x;
	voxPos = ExpandOnlyVoxPos(voxPosX);
	isMip = ((voxPosX & 0xF8000000) >> 27) != 0;
	normal = ExpandOnlyNormal(packedVoxNormalPos.y);
}

inline  __device__ ushort2 ExpandOnlyObjId(const unsigned int packVoxIdX)
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
	packed |= (uintMip & 0x0000001F) << 27;
	packed |= (voxPos.z & 0x000001FF) << 18;
	packed |= (voxPos.y & 0x000001FF) << 9;
	packed |= (voxPos.x & 0x000001FF);
	return packed;
}

inline __device__ unsigned int PackOnlyVoxNorm(const float3& normal)
{
	// (x,y components packed NORM int with 16/15 bit repectively, MSB is sign of z
	unsigned int value = 0;
	value |= __float_as_uint(normal.z) & 0x80000000;
	value |= static_cast<unsigned int>((normal.y * 0.5f + 0.5f) * 0x7FFF) << 16;
	value |= static_cast<unsigned int>((normal.x * 0.5f + 0.5f) * 0xFFFF);
	return value;
}

inline __device__ void PackVoxelNormPos(CVoxelNormPos& packedVoxNormPos,
										const uint3& voxPos,
										const float3& normal,
										const bool isMip)
{
	// First word holds span ratio and voxel position (relative to AABB or Grid)
	packedVoxNormPos.x = PackOnlyVoxPos(voxPos, isMip);

	// Second word holds normal 
	packedVoxNormPos.y = PackOnlyVoxNorm(normal);
}
#endif //__CVOXEL_H__