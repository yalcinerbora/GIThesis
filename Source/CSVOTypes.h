/**

Sparse voxel octree types that used in cuda funcs

*/

#ifndef __CSVOTYPES_H__
#define __CSVOTYPES_H__

#define GI_DENSE_WORKER_PER_PARENT 8
#define GI_SVO_WORKER_PER_NODE 8

#include <cstdint>

// Last 2 bit is unused (only slight difference then voxel pos)
typedef unsigned int CSVONode;

typedef unsigned int CSVOIrradiance; // (rgb irradiance, a component specularity)
typedef unsigned int CSVOLightDir;   // (xyz direction, w component used in average)

typedef unsigned int CSVOWeight;	 //	(xyz directional occupancy bias, w occupancy)
typedef unsigned int CSVONormal;	 // (xyz normal, w component used in average)

#pragma pack(push, 1)
struct CSVOIllumination
{
	uint64_t irradiancePortion;
	uint64_t normalPortion;
};
#pragma pack(pop)

struct CSVOLevel
{
	CSVONode *gLevelNodes;
	CSVOIllumination *gLevelIllum;
};

struct CSVOLevelConst
{
	const CSVONode *gLevelNodes;
	const CSVOIllumination *gLevelIllum;
};

//struct CSVOMaterial
//{
//	unsigned int color;		
//	unsigned int normal;
//	// Only Holding Normal and Color
//
//
//	//unsigned int props;
//	
//	// Center Nodes Has the Layout as
//	// color 8 bit each channel (RGB) A channel is empty (Pre-multiplied Alpha averaged)
//	// normal first 16 bit X, remaining 15 bit is Y, 1 bit is Z sign
//	// props first 8 bit roughness, other 8 bit is metalicity,
//	// Last 16 bit used for directional opacity
//	// 
//
//	// Leaf node has diferent values 
//	// since it will be atomically updated 
//	// color is same A channel holds count
//	// normal is same it will be updated atomiccally using count in the color component
//	// props first 16 bits same last 8 bit is count
//	// directional opacity is always opaque so its omitted
//
//	// updating structure require two updates 
//	// 64-bit atomic update for color and normal)
//	// 32-bit atomic update for material data
//};
//
//struct CSVOLeaf
//{
//};

//bool inject;
//const float3 ambientColor;

// Light Inject Related
//struct LightInjectParameters
//{
//	const float4 camPos;
//	const float3 camDir;
//
//	const CMatrix4x4* gLightVP;
//	const CLight* gLightStruct;
//
//	const float depthNear;
//	const float depthFar;
//
//	cudaTextureObject_t shadowMaps;
//	const unsigned int lightCount;
//};
#endif //__CSVOTYPES_H__