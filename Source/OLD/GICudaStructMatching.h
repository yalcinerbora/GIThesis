/**

Structure matchig between Generic System and Cuda

*/

#ifndef __GICUDASTRUCTMATCH_H__
#define __GICUDASTRUCTMATCH_H__

#include "CMatrix.cuh"
#include "CVoxelTypes.h"
#include "CAxisAlignedBB.cuh"
#include "VoxelCacheData.h"

#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"
#include "DrawBuffer.h"
#include "ThesisSolution.h"

static_assert(sizeof(IEMatrix4x4) == sizeof(CMatrix4x4), "Cuda-GL Matrix4x4 Size Mismatch.");
static_assert(sizeof(CAABB) == sizeof(AABBData), "Cuda-GL AABBData Struct Mismatch.");

static_assert(sizeof(CVoxelObjectType) == sizeof(CVoxelObjectType), "Cuda-GL VoxelType Struct Mismatch.");
static_assert(sizeof(VoxelPosition) == sizeof(VoxelNormPos), "Cuda-GL VoxelNormpos Struct Mismatch.");
//static_assert(sizeof(CVoxelIds) == sizeof(VoxelIds), "Cuda-GL VoxelIds Struct Mismatch.");
//static_assert(sizeof(CVoxelColor) == sizeof(VoxelColorData), "Cuda-GL VoxelRenderdata Struct Mismatch.");
//static_assert(sizeof(CObjectTransform) == sizeof(ModelTransform), "Cuda-GL ModelTransform Struct Mismatch.");
//static_assert(sizeof(CObjectVoxelInfo) == sizeof(ObjGridInfo), "Cuda-GL ModelTransform Struct Mismatch.");
//static_assert(sizeof(CVoxelWeight) == sizeof(VoxelWeightData), "Cuda-GL VoxelWeight Struct Mismatch.");
//static_assert(sizeof(CLight) == sizeof(Light), "Cuda-GL Light Struct Mismatch.");

#endif //__GICUDASTRUCTMATCH_H__