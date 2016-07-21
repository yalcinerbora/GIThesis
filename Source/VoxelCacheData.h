
#ifndef __VOXELCACHEDATA_H__
#define __VOXELCACHEDATA_H__

#include <cstdint>

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