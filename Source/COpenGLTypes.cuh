#pragma once
/**

Data Transfer Structs between ogl and cuda

*/

#include <cstdint>
#include "CMatrix.cuh"
#include "CAxisAlignedBB.cuh"

// Object Transform Data
// Comes from OGL Render Pipeline
#pragma pack(push, 1)
struct CObjectTransform
{
	CMatrix4x4 transform;
	CMatrix4x4 rotation;
};

struct CObjectVoxelInfo
{
	uint32_t voxelCount;
	uint32_t voxelOffset;
};
#pragma pack(pop)

typedef CAABB CObjectAABB;

struct CLight
{
	float4 position;		// position.w is the light type
	float4 direction;		// direction.w is effecting radius
	float4 color;			// color.a is intensity
};