#pragma once
/*

Global Illumination Kernels

*/

#include "CSVOTypes.h"
#include "CVoxelTypes.h"
#include <cuda.h>

class OctreeParameters;

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

//extern __global__ void GenFrontNeighborPtrs(// SVO
//											const CSVOLevel* gSVOLevels,
//											uint32_t* gLevelAllocators,
//											const uint32_t* gLevelCapacities,
//											// Limits
//											const OctreeParameters octreeParams,
//											const uint32_t nodeCount,
//											const uint32_t level);

extern __global__ void GenBackNeighborPtrs(// SVO
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