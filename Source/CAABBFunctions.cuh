#pragma once
/**

Voxel Sturcutres

*/

#include <vector_types.h>
#include "CMatrixFunctions.cuh"
#include "CVoxelTypes.h"
#include "COpenGLTypes.h"

__device__ bool Intersects(const CAABB& boxA, const CAABB& boxB);
__device__ bool CheckGridVoxIntersect(const CVoxelGrid& gGridInfo,
									  const CAABB& gObjectAABB,
									  const CMatrix4x4& gObjectTransform);

// 1.0f means use max, 0.0f means use min
__device__ static const float3 aabbLookupTable[] =
{
	{ 1.0f, 1.0f, 1.0f },		// V1
	{ 0.0f, 1.0f, 1.0f },		// V2
	{ 1.0f, 0.0f, 1.0f },		// V3
	{ 1.0f, 1.0f, 0.0f },		// V4

	{ 0.0f, 0.0f, 1.0f },		// V5
	{ 0.0f, 1.0f, 0.0f },		// V6
	{ 1.0f, 0.0f, 0.0f },		// V7
	{ 0.0f, 0.0f, 0.0f }		// V8
};

inline __device__ bool Intersects(const CAABB& boxA, const CAABB& boxB)
{
	return ((boxA.max.x > boxB.min.x) && (boxB.max.x > boxA.min.x) &&
			(boxA.max.y > boxB.min.y) && (boxB.max.y > boxA.min.y) &&
			(boxA.max.z > boxB.min.z) && (boxB.max.z > boxA.min.z));
}

inline __device__ bool CheckGridVoxIntersect(const CVoxelGrid& gGridInfo,
											 const CAABB& gObjectAABB,
											 const CMatrix4x4& gObjectTransform)
{
	// Comparing two AABB (Grid Itself is an AABB)
	const CAABB gridAABB =
	{
		{ gGridInfo.position.x, gGridInfo.position.y, gGridInfo.position.z, 1.0f },
		{
			gGridInfo.position.x + gGridInfo.span * gGridInfo.dimension.x,
			gGridInfo.position.y + gGridInfo.span * gGridInfo.dimension.y,
			gGridInfo.position.z + gGridInfo.span * gGridInfo.dimension.z,
			1.0f
		},
	};

	// Construct Transformed AABB
	CAABB transformedAABB =
	{
		{ FLT_MAX, FLT_MAX, FLT_MAX, 1.0f },
		{ -FLT_MAX, -FLT_MAX, -FLT_MAX, 1.0f }
	};

	for(unsigned int i = 0; i < 8; i++)
	{
		float3 data;
		data.x = aabbLookupTable[i].x * gObjectAABB.max.x + (1.0f - aabbLookupTable[i].x) * gObjectAABB.min.x;
		data.y = aabbLookupTable[i].y * gObjectAABB.max.y + (1.0f - aabbLookupTable[i].y) * gObjectAABB.min.y;
		data.z = aabbLookupTable[i].z * gObjectAABB.max.z + (1.0f - aabbLookupTable[i].z) * gObjectAABB.min.z;

		MultMatrixSelf(data, gObjectTransform);
		transformedAABB.max.x = fmax(transformedAABB.max.x, data.x);
		transformedAABB.max.y = fmax(transformedAABB.max.y, data.y);
		transformedAABB.max.z = fmax(transformedAABB.max.z, data.z);

		transformedAABB.min.x = fmin(transformedAABB.min.x, data.x);
		transformedAABB.min.y = fmin(transformedAABB.min.y, data.y);
		transformedAABB.min.z = fmin(transformedAABB.min.z, data.z);
	}
	bool intersects = Intersects(gridAABB, transformedAABB);
	return  intersects;
}