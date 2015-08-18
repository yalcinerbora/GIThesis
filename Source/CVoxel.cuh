/**

Voxel Sturcutres

*/

#ifndef __CVOXEL_H__
#define __CVOXEL_H__

#include <vector_types.h>

#include <vector_types.h>

// Global Voxel Data
struct CVoxelGrid
{
	float3 position;	// World Position of the voxel grid
	float span;
	uint3 dimension;	// Voxel Grid Dimentions, last component is depth of the SVO
	unsigned int depth;
};

// Main Voxel Data
typedef uint4 CVoxelPacked;

// Voxel Rendering Data
#pragma pack(push, 1)
struct CVoxelRender
{
	//unsigned int	voxelTransformType;
	uchar4			color;		// Color

	// Transform Related Data
	// For Skeletal mesh these shows index of the transforms and weights
	// For Morph target this shows the neigbouring vertices and their morph related index
	//uchar4			weightIndex;
	//uchar4			weight;
};
#pragma pack(pop)

//
__device__ inline void ExpandVoxelData(uint3& voxPos,
									   float3& normal,
									   ushort2& objId,
									   unsigned int& voxelSpanRatio,
									   unsigned int& voxRenderPtr,
									   const CVoxelPacked& packedVoxData)
{
	voxPos.x = (packedVoxData.x & 0x000001FF);
	voxPos.y = (packedVoxData.x & 0x0003FE00) >> 9;
	voxPos.z = (packedVoxData.x & 0x07FC0000) >> 18;
	voxelSpanRatio = (packedVoxData.x & 0xF8000000) >> 27;

	normal.x = (float) (packedVoxData.y & 0x0000FFFF) / 0x0000FFFF;
	normal.y = (float) ((packedVoxData.y & 0x7FFF0000) >> 16) / 0x00007FFF;
	normal.z = (((packedVoxData.y >> 31) == 1) ? -1.0f : 1.0f) * 1.0f - sqrtf(normal.x * normal.x + normal.y  * normal.y);

	objId.x = (packedVoxData.z & 0x0000FFFF);
	objId.y = (packedVoxData.z & 0xFFFF0000) >> 16;

	voxRenderPtr = packedVoxData.w;
}

__device__  inline void PackVoxelData(CVoxelPacked& packedVoxData,
									  const uint3& voxPos,
									  const float3& normal,
									  const ushort2& objId,
									  const unsigned int voxelSpanRatio,
									  const unsigned int voxRenderPtr)
{
	unsigned int value = 0;
	value |= voxelSpanRatio << 27;
	value |= voxPos.z	<< 18;
	value |= voxPos.y	<< 9;
	value |= voxPos.x;
	
	packedVoxData.x = value;

	value = 0;
	value |= signbit(normal.z) << 31;
	value |= static_cast<unsigned int>(normal.y * 0x00007FFF) << 16;
	value |= static_cast<unsigned int>(normal.x * 0x0000FFFF);
	packedVoxData.y = value;

	value = 0;
	value |= packedVoxData.y << 16;
	value |= packedVoxData.x;
	packedVoxData.z = value;

	packedVoxData.w = voxRenderPtr;
}

// This one only stores two values since objid and ptr does not change with transform matrix
__device__  inline void PackVoxelData(CVoxelPacked& packedVoxData,
									  const uint3& voxPos,
									  const float3& normal)
{
	unsigned int value = 0;
	value |= voxPos.z << 20;
	value |= voxPos.y << 10;
	value |= voxPos.x;
	packedVoxData.x = value;

	value = 0;
	value |= signbit(normal.z) << 31;
	value |= static_cast<unsigned int>(normal.y * 0x00007FFF) << 16;
	value |= static_cast<unsigned int>(normal.x * 0x0000FFFF);
	packedVoxData.y = value;
}

#endif //__CVOXEL_H__