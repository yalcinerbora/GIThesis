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
extern __device__ bool Intersects(const CAABB& boxA, const CAABB& boxB);

#endif //__CAXISALIGNEDBB_H__