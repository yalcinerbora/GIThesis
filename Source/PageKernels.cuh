#pragma once

#include <cstdint>
#include "CVoxelTypes.h"
#include "COpenGLTypes.h"


// Filter Valid Voxels
extern __global__ void FilterVoxels(// Voxel System
									CVoxelPage* gVoxelPages,
									// Dense Data from OGL
									uint32_t& gAllocator,
									uint2* gDenseData,
									uint32_t segmentOffset,
									// Limits
									uint32_t cascadeId,
									uint32_t gridSize);

extern __global__ void ClearPages(// Voxel System
								  CVoxelPage* gVoxelPages);

extern __global__ void CountVoxelsInPageSystem(uint32_t* gCounter,
											   // Voxel Cache
											   const BatchVoxelCache* gBatchVoxelCache,
											   // Voxel Pages
											   const CVoxelPageConst* gVoxelPages,
											   // Limits
											   const uint32_t batchCount);

// Initialize Pages
// Call Logic "per segment per page"
extern __global__ void InitializePage(unsigned char* emptySegments, 
									  const size_t pageCount);

// Copy valid voxels to draw OGL buffer
// Call Logic "per voxel in the grid"
extern __global__ void CopyPage(// OGL Buffer
								VoxelPosition* gVoxelPosition,
								unsigned int* gVoxelRender,
								unsigned int& gAtomicIndex,
								// Voxel Cache
								const BatchVoxelCache* gBatchVoxelCache,
								// Voxel Pages
								const CVoxelPageConst* gVoxelPages,
								// Limits
								const uint32_t batchCount,
								const uint32_t selectedCascade,
								const VoxelRenderType renderType,
								bool useCache);

// Voxel Allocate - Deallocate
// Allocates-Deallocates Voxels withn pages segment by segment
// Call Logic "per object per segement"
// Each segment allocates itself within the pages
extern __global__ void VoxelDeallocate(// Voxel System
									   CVoxelPage* gVoxelPages,
									   const CVoxelGrid* gGridInfos,
									   // Helper Structures
									   ushort2* gSegmentAllocInfo,
									   const CSegmentInfo* gSegmentInfo,
									   // Per Object Related
									   const BatchOGLData* gBatchOGLData,
									   // Limits
									   const uint32_t totalSegments);

extern __global__ void VoxelAllocate(// Voxel System
									 CVoxelPage* gVoxelPages,
									 const CVoxelGrid* gGridInfos,
									 // Helper Structures
									 ushort2* gSegmentAllocInfo,
									 const CSegmentInfo* gSegmentInfo,
									 // Per Object Related
									 const BatchOGLData* gBatchOGLData,
									 // Limits
									 const uint32_t totalSegments,
									 const uint32_t pageAmount);

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
extern  __global__ void VoxelTransform(// Voxel Pages
									   CVoxelPage* gVoxelPages,
									   const CVoxelGrid* gGridInfos,
									   // OGL Related
									   const BatchOGLData* gBatchOGLData,
									   // Voxel Cache Related
									   const BatchVoxelCache* gBatchVoxelCache,
									   // Limits
									   const uint32_t batchCount);