#pragma once
/*

Global Illumination Kernels

*/

#include "CSVOTypes.h"
#include "CVoxelTypes.h"
#include <cuda.h>

class OctreeParameters;

//// Reconstruct SVO
//// Creates SVO tree top down manner
//// For Each Level of the tree
//// First "ChildSet" then "AllocateNext" should be called
//
//// Dense version of the child set
//// Finds Dense Depth Parent and sets in on the dense 3D Array
//extern __global__ void SVOReconstructDetermineNode(CSVONode* gSVODense,
//												   const CVoxelPage* gVoxelData,
//
//												   const unsigned int cascadeNo,
//												   const CSVOConstants& svoConstants);
//
//// Sparse version of the child set
//// Finds the current level parent and traverses partially constructed tree
//// sets the child bit of the appropirate voxel
//extern __global__ void SVOReconstructDetermineNode(CSVONode* gSVOSparse,
//												   cudaTextureObject_t tSVODense,
//												   const CVoxelPage* gVoxelData,
//												   const unsigned int* gLevelOffsets,
//
//												   // Constants
//												   const unsigned int cascadeNo,
//												   const unsigned int levelDepth,
//												   const CSVOConstants& svoConstants);
//
//extern __global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,
//                                                 cudaSurfaceObject_t sDenseMat,
//
//                                                 const CSVONode* gSVODense,
//                                                 const CSVONode* gSVOSparse,
//
//                                                 const unsigned int* gLevelOffsets,
//                                                 const unsigned int& gSVOLevelOffset,
//                                                 const unsigned int& gSVONextLevelOffset,
//
//                                                 const unsigned int levelNodeCount,
//                                                 const unsigned int matOffset,
//                                                 const unsigned int currentLevel,
//                                                 const CSVOConstants& svoConstants);
//
//extern __global__ void SVOReconstructAverageNode(cudaSurfaceObject_t sDenseMatChild,
//												 cudaSurfaceObject_t sDenseMatParent,
//
//												 const unsigned int parentSize);
//

extern __global__ void AverageLevelDense(// SVO
										 const CSVOLevel& gCurrentLevel,
										 const CSVOLevelConst& gNextLevel,
										 // Limits
										 const OctreeParameters octreeParams,
										 const uint32_t currentLevelLength);

extern __global__ void AverageLevelSparse(// SVO
										  const CSVOLevel& gCurrentLevel,
										  const CSVOLevelConst& gNextLevel,
										  // Limits
										  const OctreeParameters octreeParams,
										  const uint32_t nodeCount,
										  const bool isCascadeLevel);

extern __global__ void LinkNeigbourPtrs(// SVO
										const CSVOLevel* gSVOLevels,
										uint32_t* gLevelAllocators,
										const uint32_t* gLevelCapacities,
										// Limits
										const OctreeParameters octreeParams,
										const uint32_t nodeCount,
										const uint32_t level);

extern __global__ void GenNeigbourPtrs(// SVO
									   const CSVOLevel* gSVOLevels,
									   uint32_t* gLevelAllocators,
									   const uint32_t* gLevelCapacities,									  
									   // Limits
									   const OctreeParameters octreeParams,
									   const uint32_t nodeCount,
									   const uint32_t level);

extern __global__ void AdjustIllumParameters(const CSVOLevel& gSVOLevel, uint32_t nodeCount);

extern __global__ void SVOReconstruct(// SVO
									  const CSVOLevel* gSVOLevels,
									  uint32_t* gLevelAllocators,
									  const uint32_t* gLevelCapacities,
									  // Voxel Pages
									  const CVoxelPageConst* gVoxelPages,
									  const CVoxelGrid* gGridInfos,
									  // Cache Data (for Voxel Albedo)
									  const BatchVoxelCache* gBatchVoxelCache,
									  // Inject Related									  
									  const CLightInjectParameters liParams,
									  // Limits			
									  const OctreeParameters octreeParams,
									  const uint32_t batchCount);