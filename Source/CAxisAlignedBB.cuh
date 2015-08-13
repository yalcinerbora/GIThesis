/**

Voxel Sturcutres

*/

#ifndef __CAXISALIGNEDBB_H__
#define __CAXISALIGNEDBB_H__

#include <vector_types.h>

// Global Voxel Data
// using float4 because of alignment with ogl
struct CAABB
{
	float4	min;	// World Position of the voxel grid
	float4	max;	// Voxel Grid Dimentions last component voxel span
};

__device__ inline bool Intersects(const CAABB& boxA, const CAABB& boxB)
{
	return (boxA.max.x > boxB.min.x) && (boxB.max.x > boxA.min.x) &&
			(boxA.max.y > boxB.min.y) && (boxB.max.y > boxA.min.y) &&
			(boxA.max.z > boxB.min.z) && (boxB.max.z > boxA.min.z);
}
#endif //__CAXISALIGNEDBB_H__