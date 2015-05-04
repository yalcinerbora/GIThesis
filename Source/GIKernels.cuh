/**

Global Illumination Kernels

*/

#ifndef __GIKERNELS_H__
#define __GIKERNELS_H__

#include <cuda_runtime.h>
#include <cuda.h>

struct CVoxelPacked;
struct CVoxelRender;
struct CObjectTransformOGL;
struct CVoxelGrid;
struct CObjectAABBOGL;
struct CSVONode;

// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
__global__ void VoxelTransform(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,
							   unsigned int* gEmptyMarkArray,
							   unsigned int& gEmptyMarkIndex,
							   const CObjectTransformOGL* gObjTransforms,
							   const CVoxelGrid& gGridInfo);


// Voxel Introduce
// Introduces existing voxel to the voxel grid
// Call Logic "per voxel in an object"
__global__ void VoxelIntroduce(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,
							   unsigned int* gEmptyMarkArray,
							   unsigned int& gEmptyMarkIndex,
							   const CVoxelPacked* gObjectVoxelCache,
							   const CVoxelRender* gObjectVoxelRenderCache,
							   const CObjectTransformOGL& gObjTransform,
							   const CVoxelGrid& gGridInfo));

// Voxel Introduce Helper Function
// Object Cull
// Determines that this object should enter "VoxelIntroduce" Kernel
// Call Logic "per object"
__global__ void VoxelObjectCull(unsigned int* gObjectIndices,
								unsigned int& gIndicesIndex,
								const CObjectAABBOGL* gObjectAABB,
								const CObjectTransformOGL* gObjTransforms,
								const CVoxelGrid& gGridInfo);

// Reconstruct SVO
// Creates SVO tree top down manner
// Implementation is opposite of parallel reduction
// Call Logic "per svo node (varying)"
__global__ void SVOReconstruct(CSVONode* svo,
	
								const CVoxelPacked* gVoxelData,

							   );


// Voxelize
// Do Voxelization of a mesh
// Its better to use OGL here since we render triangles to a array




#endif //__GIKERNELS_H__