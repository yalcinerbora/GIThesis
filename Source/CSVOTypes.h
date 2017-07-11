#pragma once
/**

Sparse voxel octree types that used in cuda funcs

*/
#define GI_DENSE_WORKER_PER_PARENT 8
#define GI_SVO_NODE_WORK_PER_THREAD 8

#include <cstdint>
#include "CVoxelTypes.h"

struct CLight;
struct CMatrix4x4;

// Last 2 bit is unused (only slight difference then voxel pos)
//typedef unsigned int CSVONode;
struct CSVONode
{
	uint32_t next;
	uint32_t neigborus[3];
};

// If i put CSVO node into packed pragma it fails on runtime (SVO Reconsrtuct kernel)
// I've no idea why
static_assert(sizeof(CSVONode) == sizeof(uint32_t) * 4, "CSVO Node should be packed");

typedef unsigned int CSVOIrradiance; // (rgb irradiance, a component specularity)
typedef unsigned int CSVOLightDir;   // (xyz direction, w component used in average)

typedef unsigned int CSVOWeight;	 //	(xyz directional occupancy bias, w occupancy)
typedef unsigned int CSVONormal;	 // (xyz normal, w component used in average)

#pragma pack(push, 1)
struct CSVOIllumination
{
	uint64_t irradiancePortion;	// LS 32-bit irradiance, other normal
	uint64_t occupancyPortion;	// LS 32-bit occupancy, other lightdir
};
#pragma pack(pop)

struct CSVOLevel
{
	CSVONode* gLevelNodes;
	CSVOIllumination* gLevelIllum;
};

struct CSVOLevelConst
{
	const CSVONode* gLevelNodes;
	const CSVOIllumination* gLevelIllum;
};

// Light Inject Related
struct CLightInjectParameters
{
	const float3			ambientLight;

	const bool				injectOn;
	const float4			camPos;
	const float3			camDir;

	const CMatrix4x4*		gLightVP;
	const CLight*			gLightStruct;

	// Shadow map related
	const float				depthNear;
	const float				depthFar;
	cudaTextureObject_t		shadowMaps;
	const uint32_t			lightCount;
};