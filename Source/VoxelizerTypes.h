#pragma once

#include <cstdint>
#include "StructuredBuffer.h"

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