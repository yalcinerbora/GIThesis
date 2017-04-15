
#ifndef __VOXELCACHEDATA_H__
#define __VOXELCACHEDATA_H__

#include <cstdint>
#include "StructuredBuffer.h"

enum class VoxelObjectType
{
	STATIC,			// Object does not move
	DYNAMIC,		// Object does move (with transform matrices)
	SKEL_DYNAMIC,	// Object moves with weighted transformation matrices
	MORPH_DYNAMIC,	// Object moves with morph targets (each voxel has their adjacent vertex morphs weighted)
};

#pragma pack(push, 1)
struct VoxelNormPos
{
	uint32_t vNormPos[2];
};

struct VoxelIds
{
	uint32_t vIds[2];
};

typedef uint32_t VoxelColorData;

struct VoxelWeightData
{
	uint32_t		weight;
	uint32_t		weightIndex;
};
#pragma pack(pop)
#endif //__VOXELCACHEDATA_H__