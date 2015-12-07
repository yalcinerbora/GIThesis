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
struct CVoxelRender
{
	//unsigned int	voxelTransformType;
	uchar4			color;		// Color

	// Transform Related Data
	// For Skeletal mesh these shows index of the transforms and weights
	// For Morph target this shows the neigbouring vertices and their morph related index
	//uchar4		weightIndex;
	//uchar4		weight;
};

struct CVoxelRenderSkelMorph
{
	//unsigned int	voxelTransformType;
	uchar4			color;		// Color

	// Transform Related Data
	// For Skeletal mesh these shows index of the transforms and weights
	// For Morph target this shows the neigbouring vertices and their morph related index
	uchar4			weightIndex;
	uchar4			weight;
};
#pragma pack(pop)
#endif //__CVOXELTYPES_H__