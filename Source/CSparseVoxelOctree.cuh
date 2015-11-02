/**

Sparse voxel octree implementation
Designed for fast reconstruction from its bottom 

*/

#ifndef __CSPARSEVOXELOCTREE_H__
#define __CSPARSEVOXELOCTREE_H__

#include <cuda_runtime.h>
#include <cuda.h>
#include "CVoxel.cuh"

// first int has
// first 24 bit is children index
// last 8 bit used to determine which children is avail
// --
// last 4 byte is used for color
typedef unsigned int CSVONode;
typedef unsigned int CSVOColor;

inline __device__ void UnpackNode(unsigned int& childrenIndex,
								  unsigned char& childrenBit,
								  const CSVONode& node)
{
	childrenIndex = node & 0x00FFFFFF;
	childrenBit = node >> 24;
}

inline __device__ CSVONode PackNode(const unsigned int& childrenIndex,
									const unsigned char& childrenBit)
{
	CSVONode node;
	node = childrenBit << 24;
	node |= childrenIndex;
	return node;
}

inline __device__ float4 UnpackSVOColor(const CSVOColor& node)
{
	float4 color;
	color.x = static_cast<float>((node & 0x000000FF) >> 0) / 255.0f;
	color.y = static_cast<float>((node & 0x0000FF00) >> 8) / 255.0f;
	color.z = static_cast<float>((node & 0x00FF0000) >> 16) / 255.0f;
	color.w = static_cast<float>((node & 0xFF000000) >> 24);
	return color;
}

inline __device__ CSVOColor PackSVOColor(const float4& color)
{
	CSVOColor colorPacked;
	colorPacked = static_cast<unsigned int>(color.w) << 24;
	colorPacked |= static_cast<unsigned int>(color.z * 255.0f) << 16;
	colorPacked |= static_cast<unsigned int>(color.y * 255.0f) << 8;
	colorPacked |= static_cast<unsigned int>(color.x * 255.0f) << 0;
	return colorPacked;
}

// Returns the intersected voxelIndex if voxel is found
// Else it returns -1 (0xFFFFFFFF)
inline __device__ unsigned int ConeTrace() {return 0;}
#endif //__CSPARSEVOXELOCTREE_H__