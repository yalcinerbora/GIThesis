/**

Global Illumination Kernels

*/

#ifndef __GIKERNELS_H__
#define __GIKERNELS_H__

#include "GICudaAllocator.h"
#include "CSVOTypes.cuh"

struct CAABB;
typedef CAABB CObjectAABB;
typedef uint2 CVoxelNormPos;
typedef uint2 CVoxelIds;
struct CObjectTransform;
struct CVoxelColor;
struct CVoxelPage;
struct CVoxelGrid;
struct CSVOConstants;

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
extern  __global__ void VoxelTransform(// Voxel Pages
									   CVoxelPage* gVoxelData,
									   const CVoxelGrid& gGridInfo,
									   const float3 hNewGridPosition,

									   // Object Related
									   CObjectTransform** gObjTransforms,
									   uint32_t** gObjTransformIds,
									   CVoxelNormPos** gVoxNormPosCacheData,
									   CVoxelColor** gVoxRenderData,
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
										  const CObjectTransform* gObjTransforms,
										  const unsigned int* gObjTransformIds);

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
										const CObjectTransform* gObjTransforms,
										const unsigned int* gObjTransformIds);

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

										  // Per Voxel Related
										  const CVoxelIds* gVoxelIdsCache,
										  uint32_t voxCount,
										  uint32_t objCount,

										  // Batch(ObjectGroup in terms of OGL) Id
										  uint32_t batchId);


// Reconstruct SVO
// Creates SVO tree top down manner
// For Each Level of the tree
// First "ChildSet" then "AllocateNext" should be called

// Dense version of the child set
// Finds Dense Depth Parent and sets in on the dense 3D Array
extern __global__ void SVOReconstructDetermineNode(CSVONode* gSVODense,
												   const CVoxelPage* gVoxelData,

												   const unsigned int cascadeNo,
												   const CSVOConstants& svoConstants);

// Sparse version of the child set
// Finds the current level parent and traverses partially constructed tree
// sets the child bit of the appropirate voxel
extern __global__ void SVOReconstructDetermineNode(CSVONode* gSVOSparse,
												   cudaTextureObject_t tSVODense,
												   const CVoxelPage* gVoxelData,
												   const unsigned int* gLevelOffsets,

												   // Constants
												   const unsigned int cascadeNo,
												   const unsigned int levelDepth,
												   const CSVOConstants& svoConstants);

// Allocate next alloates the next level of the tree
extern __global__ void SVOReconstructAllocateLevel(CSVONode* gSVOLevel,
												   unsigned int& gSVONextLevelAllocator,
												   const unsigned int& gSVONextLevelTotalSize,
												   const unsigned int& gSVOLevelTotalSize,
												   const CSVOConstants& svoConstants);

extern __global__ void SVOReconstructMaterialLeaf(CSVOMaterial* gSVOMat,

												  // Const SVO Data
												  const CSVONode* gSVOSparse,
												  const unsigned int* gLevelOffsets,
												  cudaTextureObject_t tSVODense,

												  // Page Data
												  const CVoxelPage* gVoxelData,

												  // For Color Lookup
												  CVoxelColor** gVoxelRenderData,

												  // Constants
												  const unsigned int matSparseOffset,
												  const unsigned int cascadeNo,
												  const CSVOConstants& svoConstants);

extern __global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,
												 cudaSurfaceObject_t sDenseMat,

												 const CSVONode* gSVODense,
												 const CSVONode* gSVOSparse,

												 const unsigned int* gLevelOffsets,
												 const unsigned int& gSVOLevelOffset,
												 const unsigned int& gSVONextLevelOffset,

												 const unsigned int levelNodeCount,
												 const unsigned int matOffset,
												 const unsigned int currentLevel,
												 const CSVOConstants& svoConstants);

extern __global__ void SVOReconstructAverageNode(cudaSurfaceObject_t sDenseMatChild,
												 cudaSurfaceObject_t sDenseMatParent,

												 const unsigned int parentSize);

extern __global__ void SVOReconstruct(CSVOMaterial* gSVOMat,
									  CSVONode* gSVOSparse,
									  CSVONode* gSVODense,
									  unsigned int* gLevelAllocators,

									  const unsigned int* gLevelOffsets,
									  const unsigned int* gLevelTotalSizes,
									  
									  // For Color Lookup
									  const CVoxelPage* gVoxelData,
									  CVoxelColor** gVoxelRenderData,

									  const unsigned int matSparseOffset,
									  const unsigned int cascadeNo,
									  const CSVOConstants& svoConstants);


#endif //__GIKERNELS_H__