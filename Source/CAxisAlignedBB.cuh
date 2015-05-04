/**

Voxel Sturcutres

*/

#ifndef __CAXISALIGNEDBB_H__
#define __CAXISALIGNEDBB_H__

#include <vector_types.h>

// Global Voxel Data
struct CAABB
{
	float3	min;	// World Position of the voxel grid
	float3	max;	// Voxel Grid Dimentions last component voxel span
};

__device__ inline  bool Intersects(const CAABB& boxA, const CAABB& boxB)
{
	bool result = true;
	result &= (boxA.max.x >= boxB.min.x) && (boxB.max.x >= boxA.min.x);	
	result &= (boxA.max.y >= boxB.min.y) && (boxB.max.y >= boxA.min.y);
	result &= (boxA.max.z >= boxB.min.z) && (boxB.max.z >= boxA.min.z);
	return result;
}
#endif //__CAXISALIGNEDBB_H__