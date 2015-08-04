/**

Global Illumination Kernels

*/

#ifndef __GIKERNELS_H__
#define __GIKERNELS_H__

#include "GICudaAllocator.h"

struct CAABB;
typedef CAABB CObjectAABB;
typedef uint4 CVoxelPacked;
struct CObjectTransform;
struct CVoxelRender;
struct CVoxelPage;
struct CVoxelGrid;
struct CSVONode;

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
extern  __global__ void VoxelTransform(// Voxel Pages
									   CVoxelPage* gVoxelData,
									   CVoxelGrid& gGridInfo,
									   const float3& gNewGridPosition,

									   // Per Object Segment
									   ushort2** gObjectAllocLocations,

									   // Object Related
									   const unsigned int** gObjectAllocIndexLookup,
									   const CObjectTransform** gObjTransformsRelative);

// Voxel Introduce
// Introduces existing voxel to the voxel grid
// Call logic changes internally
// Call Logic "per object per segement"
// Each segment allocates itself within the pages
// Call Logic "per voxel"
// Each voxel writes its data to allocated segments
extern __global__ void VoxelObjectInclude(// Voxel System
										  CVoxelPage* gVoxelData,
										  const unsigned int gPageAmount,
										  const CVoxelGrid& gGridInfo,

										  // Per Object Segment Related
										  ushort2* gObjectAllocLocations,
										  unsigned int* gSegmentObjectId,
										  uint32_t totalSegments,

										  // Per Object Related
										  char* gWriteToPages,
										  const unsigned int* gObjectVoxStrides,
										  const unsigned int* gObjectAllocIndexLookup,
										  const CObjectAABB* gObjectAABB,
										  const CObjectTransform* gObjTransforms,
										  const CObjectVoxelInfo* gObjInfo,
										  uint32_t objectCount,

										  // Per Voxel Related
										  const CVoxelPacked* gObjectVoxelCache,
										  uint32_t voxCount,

										  // Batch(ObjectGroup in terms of OGL) Id
										  uint32_t batchId);

// Object Exlude
// Determines that this object's segments should deallocated
// Call Logic "per object per segement"
extern __global__ void VoxelObjectExclude(// Voxel System
										  CVoxelPage* gVoxelData,
										  const unsigned int gPageAmount,
										  const CVoxelGrid& gGridInfo,

										  // Per Object Segment Related
										  ushort2* gObjectAllocLocations,
										  unsigned int* gSegmentObjectId,
										  uint32_t totalSegments,

										  // Per Object Related
										  const CObjectAABB* gObjectAABB,
										  const CObjectTransform* gObjTransforms);

// Reconstruct SVO
// Creates SVO tree top down manner
// Implementation is opposite of parallel reduction
// Call Logic "per svo node (varying)"
extern  __global__ void SVOReconstruct(CSVONode* svo,
									   const CVoxelPacked** gVoxelData);

#endif //__GIKERNELS_H__