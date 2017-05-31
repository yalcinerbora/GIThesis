#pragma once
/**

Data Transfer Structs between ogl and cuda

*/

#include <cstdint>
#include <vector_types.h>
#include "SceneLights.h"
#include "Globals.h"

#pragma pack(push, 1)
struct CAABB
{
	float4	min;	// World Position of the voxel grid
	float4	max;	// Voxel Grid Dimentions last component voxel span
};

struct CMatrix4x4
{
	float4 column[4];
};

struct CMatrix3x3
{
	float4 column[3];
};

struct CModelTransform
{
	CMatrix4x4 transform;
	CMatrix4x4 rotation;
};
#pragma pack(pop)

typedef CModelTransform CJointTransform;

struct CLight
{
	float4 position;	// position.w is the light type
	float4 direction;	// direction.w is effecting radius
	float4 color;		// color.a is intensity
};

// Static Assertions
static_assert(sizeof(CAABB) == sizeof(AABBData), 
			  "CUDA CAABB data do not have the same size as renerer AABBData.");
static_assert(sizeof(CMatrix4x4) == sizeof(IEMatrix4x4),
			  "CUDA CMatrix4x4 data do not have the same size as renderer IEMatrix4x4.");
static_assert(sizeof(CLight) == sizeof(Light),
			  "CUDA CLight data do not have the same size as renderer Light.");