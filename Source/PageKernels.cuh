#pragma once

#include <cstdint>
#include "CVoxelTypes.h"
#include "GIVoxelPages.h"
//
//struct CAABB;
//typedef CAABB CObjectAABB;
//typedef uint2 CVoxelNormPos;
//typedef uint2 CVoxelIds;
//
//struct CObjectTransform;
//struct CVoxelColor;
//struct CVoxelPage;
//struct CVoxelGrid;
//struct CSVOConstants;

// Initialize Pages
extern __global__ void InitializePage(unsigned char* emptySegments,
									  const ptrdiff_t stride,
									  const size_t pageCount);

// Voxel Allocate - Deallocate
// Allocates-Deallocates Voxels withn pages segment by segment
// Call Logic "per object per segement"
// Each segment allocates itself within the pages
extern __global__ void VoxelObjectDealloc(// Voxel System
										  CVoxelPage* gVoxelData,
										  const CVoxelGrid& gGridInfo,

										  // Per Object Segment Related
										  ushort2* gObjectAllocLocations,
										  const SegmentObjData* gSegmentObjectData,
										  const uint32_t totalSegments,

										  // Per Object Related
										  const CObjectAABB* gObjectAABB,
										  const CObjectTransform* gObjTransforms,
										  const uint32_t* gObjTransformIds);

extern __global__ void VoxelObjectAlloc(// Voxel System
										CVoxelPage* gVoxelData,
										const unsigned int gPageAmount,
										const CVoxelGrid& gGridInfo,

										// Per Object Segment Related
										ushort2* gObjectAllocLocations,
										const SegmentObjData* gSegmentObjectData,
										const uint32_t totalSegments,

										// Per Object Related
										const CObjectAABB* gObjectAABB,
										const CObjectTransform* gObjTransforms,
										const uint32_t* gObjTransformIds);

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
extern  __global__ void VoxelTransform(// Voxel Pages
									   CVoxelPage* gVoxelData,
									   const CVoxelGrid& gGridInfo,
									   const float3 hNewGridPosition,

									   // Object Related
									   CObjectTransform** gObjTransforms,
									   CObjectTransform** gJointTransforms,
									   uint32_t** gObjTransformIds,
									   CVoxelNormPos** gVoxNormPosCacheData,
									   CVoxelColor** gVoxRenderData,
									   CVoxelWeight** gVoxWeightData,

									   CObjectVoxelInfo** gObjInfo,
									   CObjectAABB** gObjectAABB);

// Voxel Clear Marked
// Clears the deallocated voxels marked by "VxoelObjecDealloc" function
// Logic per voxel in page system
__global__ void VoxelClearMarked(CVoxelPage* gVoxelData);

// Voxel Clear Signal
// Stops Clear Signal 
// Logic per segment in page system
__global__ void VoxelClearSignal(CVoxelPage* gVoxelData,
								 const uint32_t numPages);