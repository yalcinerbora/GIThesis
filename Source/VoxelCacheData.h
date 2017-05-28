
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

typedef uint32_t VoxelPosition;
typedef uint32_t VoxelNormal;
typedef uint32_t VoxelAlbedo;

#pragma pack(push, 1)
struct VoxelWeights
{
	uint32_t		weight;
	uint32_t		weightIndex;
};

struct MeshVoxelInfo
{
	uint32_t voxCount;
	uint32_t voxOffset;
};
#pragma pack(pop)
#endif //__VOXELCACHEDATA_H__