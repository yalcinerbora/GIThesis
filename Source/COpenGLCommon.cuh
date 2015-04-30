

/**

Data Transfer Structs between ogl and cuda

*/

#ifndef __COPENGLCOMMON_H__
#define __COPENGLCOMMON_H__

#include "CMatrix.cuh"

/// Object Transform Data
// Comes from OGL Render Pipeline
struct CObjectTransformOGL
{
	CMatrix4x4 transform;
	CMatrix3x3 rotation;
};

struct CObjectAABBOGL
{
	float3 min;
	float3 max;
};

#endif //__COPENGLCOMMON_H__