/**



*/

#ifndef __CVOXELTYPES_H__
#define __CVOXELTYPES_H__

#include <vector_types.h>
#include <cstdint>

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

// Further Seperated Voxel Data
typedef unsigned int CVoxelPos;
typedef unsigned int CVoxelNorm;
typedef unsigned int CVoxelOccupancy;
typedef uchar4 CVoxelAlbedo;

// Voxel Rendering Data
#pragma pack(push, 1)
struct CVoxelWeight
{
	uchar4			weight;
	uchar4			weightIndex;
};
#pragma pack(pop)

enum class SegmentOccupation : unsigned char
{
	EMPTY = 0,
	OCCUPIED = 1,
	MARKED_FOR_CLEAR = 2,
};

struct SegmentObjData
{
	uint16_t			batchId;
	uint16_t			objId;
	uint16_t			objectSegmentId;
	uint16_t			packed;	// Containts 2 bit Obj Type 4 bit Occupation 10 bit segment occupancy
	uint32_t			voxStride;
};

struct CVoxelPage
{
	CVoxelPos*			dGridVoxPos;
	CVoxelNorm*			dGridVoxNorm;
	CVoxelOccupancy*	dGridVoxOccupancy;
	unsigned char*		dEmptySegmentPos;
	SegmentObjData*		dSegmentObjData;
	unsigned int		dEmptySegmentStackSize;
};
#endif //__CVOXELTYPES_H__