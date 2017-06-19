//#pragma once
///**
//
//Global Illumination Kernels
//
//*/
//
//#include "CSVOTypes.h"
//#include "CVoxelTypes.h"
//
//
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
//extern __global__ void SVOReconstruct(CSVOMaterial* gSVOMat,
//                                      CSVONode* gSVOSparse,
//                                      CSVONode* gSVODense,
//                                      unsigned int* gLevelAllocators,
//
//                                      const unsigned int* gLevelOffsets,
//                                      const unsigned int* gLevelTotalSizes,
//
//                                      // For Color Lookup
//                                      const CVoxelPage* gVoxelData,
//                                      CVoxelAlbedo** gVoxelRenderData,
//
//                                      const unsigned int matSparseOffset,
//                                      const unsigned int cascadeNo,
//                                      const CSVOConstants& svoConstants,
//
//                                      // Light Inject Related
//                                      bool inject,
//                                      float span,
//                                      const float3 outerCascadePos,
//                                      const float3 ambientColor,
//
//                                      const float4 camPos,
//                                      const float3 camDir,
//
//                                      //const CMatrix4x4* lightVP,
//                                      //const CLight* lightStruct,
//
//                                      const float depthNear,
//                                      const float depthFar,
//
//                                      cudaTextureObject_t shadowMaps,
//                                      const unsigned int lightCount);