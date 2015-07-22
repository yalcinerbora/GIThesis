

/**

Data Transfer Structs between ogl and cuda

*/

#ifndef __COPENGLCOMMON_H__
#define __COPENGLCOMMON_H__

#include "CMatrix.cuh"
#include "CAxisAlignedBB.cuh"

// Object Transform Data
// Comes from OGL Render Pipeline
struct CObjectTransform
{
	CMatrix4x4 transform;
	CMatrix4x4 rotation;
};

typedef CAABB CObjectAABB;

struct CObjectVoxelInfo
{
	float span;
	unsigned int voxelCount;
};

#endif //__COPENGLCOMMON_H__