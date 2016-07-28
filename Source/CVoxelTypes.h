/**



*/

#ifndef __CVOXELTYPES_H__
#define __CVOXELTYPES_H__

#include <vector_types.h>

enum class CVoxelObjectType
{
	STATIC,			// Object does not move
	DYNAMIC,		// Object does move (with transform matrices)
	SKEL_DYNAMIC,	// Object moves with weighted transformation matrices
	MORPH_DYNAMIC,	// Object moves with morph targets (each voxel has their adjacent vertex morphs weighted)
};

// Global Voxel Data
struct CVoxelGrid
{
	float3			position;	// World Position of the voxel grid
	float			span;
	uint3			dimension;	// Voxel Grid Dimentions
	unsigned int	depth;
};

// Seperated Voxel Data
typedef uint2 CVoxelNormPos;
typedef uint2 CVoxelIds;

// Further Seperated Voxel Data
typedef unsigned int CVoxelPos;
typedef unsigned int CVoxelNorm;

// Voxel Rendering Data
#pragma pack(push, 1)
struct CVoxelColor
{
	//unsigned int	voxelTransformType;
	uchar4			color;		// Color
};

struct CVoxelWeight
{
	uchar4			weight;
	uchar4			weightIndex;
};
#pragma pack(pop)
#endif //__CVOXELTYPES_H__