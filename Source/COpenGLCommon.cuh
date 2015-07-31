

/**

Data Transfer Structs between ogl and cuda

*/

#ifndef __COPENGLCOMMON_H__
#define __COPENGLCOMMON_H__

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
	float span;
	unsigned int voxelCount;
};
#pragma pack(pop)

typedef CAABB CObjectAABB;

#endif //__COPENGLCOMMON_H__