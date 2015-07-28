

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

#define GI_CUDA_OBJ_STATE_IN 0
#define GI_CUDA_OBJ_STATE_ADD 0
#define GI_CUDA_OBJ_STATE_REMOVE 0
#define GI_CUDA_OBJ_STATE_OUT 1

#endif //__COPENGLCOMMON_H__