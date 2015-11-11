/**

Global Illumination Kernels

*/

#ifndef __GIKERNELS_H__
#define __GIKERNELS_H__

#include "GICudaAllocator.h"

struct CAABB;
typedef CAABB CObjectAABB;
typedef uint2 CVoxelNormPos;
typedef uint2 CVoxelIds;
struct CObjectTransform;
struct CVoxelRender;
struct CVoxelPage;
struct CVoxelGrid;
struct CSVOConstants;
typedef unsigned int CSVONode;

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
extern  __global__ void VoxelTransform(// Voxel Pages
									   CVoxelPage* gVoxelData,
									   const CVoxelGrid& gGridInfo,
									   const float3 hNewGridPosition,

									   // Per Object Segment
									   ushort2** gObjectAllocLocations,

									   // Object Related
									   unsigned int** gObjectAllocIndexLookup,
									   CObjectTransform** gObjTransforms,
									   CVoxelNormPos** gVoxNormPosCacheData,
									   CVoxelRender** gVoxRenderData,
									   CObjectVoxelInfo** gObjInfo,
									   CObjectAABB** gObjectAABB);

// Voxel Allocate - Deallocate
// Allocates-Deallocates Voxels withn pages segment by segment
// Call Logic "per object per segement"
// Each segment allocates itself within the pages
extern __global__ void VoxelObjectDealloc(// Voxel System
										  CVoxelPage* gVoxelData,
										  const CVoxelGrid& gGridInfo,

										  // Per Object Segment Related
										  ushort2* gObjectAllocLocations,
										  const unsigned int* gSegmentObjectId,
										  const uint32_t totalSegments,

										  // Per Object Related
										  char* gWriteToPages,
										  const CObjectAABB* gObjectAABB,
										  const CObjectTransform* gObjTransforms);

extern __global__ void VoxelObjectAlloc(// Voxel System
										CVoxelPage* gVoxelData,
										const unsigned int gPageAmount,
										const CVoxelGrid& gGridInfo,

										// Per Object Segment Related
										ushort2* gObjectAllocLocations,
										const unsigned int* gSegmentObjectId,
										const uint32_t totalSegments,

										// Per Object Related
										char* gWriteToPages,
										const CObjectAABB* gObjectAABB,
										const CObjectTransform* gObjTransforms);

// Voxel Clear Marked
// Clears the deallocated voxels marked by "VxoelObjecDealloc" function
// Logic per voxel in page system
__global__ void VoxelClearMarked(CVoxelPage* gVoxelData);

// Voxel Clear Signal
// Stops Clear Signal 
// Logic per segment in page system
__global__ void VoxelClearSignal(CVoxelPage* gVoxelData,
								 const uint32_t numPages);

// Voxel Include
// Introduces existing voxel to the voxel grid
// Call Logic "per voxel"
// Each voxel writes its data to allocated segments
extern __global__ void VoxelObjectInclude(// Voxel System
										  CVoxelPage* gVoxelData,
										  const CVoxelGrid& gGridInfo,

										  // Per Object Segment Related
										  ushort2* gObjectAllocLocations,
										  const uint32_t segmentCount,
										  
										  // Per Object Related
										  char* gWriteToPages,
										  const unsigned int* gObjectVoxStrides,
										  const unsigned int* gObjectAllocIndexLookup,
										  const CObjectAABB* gObjectAABB,
										  const CObjectTransform* gObjTransforms,
										  const CObjectVoxelInfo* gObjInfo,

										  // Per Voxel Related
										  const CVoxelIds* gVoxelIdsCache,
										  uint32_t voxCount,

										  // Batch(ObjectGroup in terms of OGL) Id
										  uint32_t batchId);


// Reconstruct SVO
extern __global__ void SVOReconstruct(CSVONode* gSVOSparse,
									  CSVONode* gSVODense,
									  unsigned int* gSVOLock,

									  // SVO Alloc Location Holding Data
									  unsigned int* gSVOEmptyLoc,
									  unsigned int& gSVOLocIndex,
									  const unsigned int maxSVOCount,

									  // Voxel Page Data
									  const CVoxelPage* gVoxelData,

									  const unsigned int cascadeNo,
									  const CSVOConstants& svoConstants);
#endif //__GIKERNELS_H__