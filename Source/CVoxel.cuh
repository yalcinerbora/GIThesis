/**

Voxel Sturcutres

*/

#ifndef __CVOXEL_H__
#define __CVOXEL_H__

#include <vector_types.h>

// Global Voxel Data
struct CVoxelGrid
{
	float3 position;	// World Position of the voxel grid last component voxel span
	float span;
	uint3 dimension;	// Voxel Grid Dimentions, last component is depth of the SVO
	unsigned int depth;
};

// Main Voxel Data
typedef uint2 CVoxelPacked;

// Voxel Rendering Data
#pragma pack(push, 1)
struct CVoxelRender
{
	float3 normal;		// World Normal
	uchar4 color;		// Color

	// Add transofrm related data (if skeletal mesh, or morph target mesh)


};
#pragma pack(pop)

//
__device__ inline void ExpandVoxelData(uint3& voxPos, 
									   unsigned int& objId,
									   const CVoxelPacked& packedVoxData)
{
	voxPos.x	= (packedVoxData.x && 0x0000FFFF);
	voxPos.y	= (packedVoxData.x && 0xFFFF0000) >> 16;
	voxPos.z	= (packedVoxData.y && 0x0000FFFF);
	objId		= (packedVoxData.y && 0xFFFF0000) >> 16;
}

__device__  inline void PackVoxelData(CVoxelPacked& packedVoxData,
									  const uint3& voxPos,
									  const unsigned int& objId)
{
	packedVoxData.x  = voxPos.x;
	packedVoxData.x |= voxPos.y << 16;
	packedVoxData.y  = voxPos.z;
	packedVoxData.y |= objId	<< 16;
}

#endif //__CVOXEL_H__