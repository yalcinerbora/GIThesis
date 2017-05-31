#pragma once

#include <cstdint>
#include "CVoxelTypes.h"
#include "COpenGLTypes.h"

// Initialize Pages
extern __global__ void InitializePage(unsigned char* emptySegments, 
									  const size_t pageCount);

// Voxel Allocate - Deallocate
// Allocates-Deallocates Voxels withn pages segment by segment
// Call Logic "per object per segement"
// Each segment allocates itself within the pages
extern __global__ void VoxelObjectIO(// Voxel System
									 CVoxelPage* gVoxelData,
									 const CVoxelGrid* gGridInfos,
									 // Helper Structures
									 ushort2* gSegmentAllocInfo,
									 const CSegmentInfo* gSegmentInfo,
									 // Per Object Related
									 const BatchOGLData* gBatchOGLData,
									 // Limits
									 const uint32_t totalSegments,
									 const uint32_t pageAmount);

//extern __global__ void VoxelObjectAlloc(// Voxel System
//										CVoxelPage* gVoxelData,
//										const CVoxelGrid* gGridInfos,
//										// Helper Structures
//										ushort2* gSegmentAllocInfo,
//										const CSegmentInfo* gSegmentInfo,
//										// Per Object Related
//										const BatchOGLData* gBatchOGLData,
//										// Limits
//										const uint32_t totalSegments,
//										const uint32_t pageAmount);

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
extern  __global__ void VoxelTransform(// Voxel Pages
									   CVoxelPage* gVoxelPages,
									   const CVoxelGrid* gGridInfos,
									   const float3* gNewGridPositions,
									   // OGL Related
									   const BatchOGLData* gBatchOGLData,
									   // Voxel Cache Related
									   const BatchVoxelCache* gBatchVoxelCache,
									   // Limits
									   const uint32_t batchCount);

// Voxel Clear Marked
// Clears the deallocated voxels marked by "VxoelObjecDealloc" function
// Logic per voxel in page system
__global__ void VoxelClearMarked(CVoxelPage* gVoxelData);

// Voxel Clear Signal
// Stops Clear Signal 
// Logic per segment in page system
__global__ void VoxelClearSignal(CVoxelPage* gVoxelData,
								 const uint32_t numPages);


//__host__ void