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
struct CVoxelRender
{
	float3 normal;		// World Normal
	uchar4 color;		// Color
};

//
__device__ void ExpandVoxelData(uint3& voxPos, unsigned int& objID, 
								const CVoxelPacked& packedVoxData);
__device__ void PackVoxelData(CVoxelPacked& packedVoxData,
							  const uint3& voxPos, 
							  const unsigned int& objID);

#endif //__CVOXEL_H__