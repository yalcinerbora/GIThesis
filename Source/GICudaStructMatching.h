/**

Structure matchig between Generic System and Cuda

*/

#ifndef __GICUDASTRUCTMATCH_H__
#define __GICUDASTRUCTMATCH_H__

#include "CMatrix.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"

#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"
#include "DrawBuffer.h"
#include "ThesisSolution.h"

static_assert(sizeof(IEMatrix4x4) == sizeof(CMatrix4x4), "Cuda-GL Matrix4x4 Size Mismatch.");
static_assert(sizeof(CAABB) == sizeof(AABBData), "Cuda-GL AABBData Struct Mismatch.");

static_assert(sizeof(CVoxelPacked) == sizeof(VoxelData), "Cuda-GL VoxelData Struct Mismatch.");
static_assert(sizeof(CVoxelRender) == sizeof(VoxelRenderData), "Cuda-GL VoxelRenderdata Struct Mismatch.");
static_assert(sizeof(CObjectTransform) == sizeof(ModelTransform), "Cuda-GL ModelTransform Struct Mismatch.");
static_assert(sizeof(CObjectVoxelInfo) == sizeof(ObjGridInfo), "Cuda-GL ModelTransform Struct Mismatch.");

#endif //__GICUDASTRUCTMATCH_H__