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
struct CObjectAABB;


// Voxel Transform
// Transforms existing voxels in order to cut voxel reconstruction each frame
// Call Logic "per voxel in the grid"
__global__ void VoxelTransform(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,
							   const CObjectTransformOGL* gObjTransforms,
							   const CVoxelGrid& globalVoxel);


// Voxel Introduce
// Introduces existing voxel to the voxel grid
// Call Logic "per object per voxel" (object batched)
__global__ void VoxelIntroduce(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,

							   // Teoretical Input
							   const CVoxelPacked** gVoxelCache,
							   const CVoxelRender** gVoxelRenderCache,
							   const CObjectTransformOGL* gObjTransforms,
							   const unsigned int* gObjectIndices,
							   const CVoxelGrid& globalGridInfo);

// Voxel Introduce Helper Function
// Object Cull
// Determines that this object should enter "VoxelIntroduce" Kernel
// Call Logic "per object"
__global__ void VoxelObjectCull(unsigned int* gObjectIndices,
								unsigned int gIndicesIndex,
								const CObjectAABB* gObjectAABB,
								const CObjectTransformOGL* gObjTransforms,
								const CVoxelGrid& globalGridInfo)
{
	unsigned int globalId;
	unsigned int writeIndex;



	// Compare Transformed AABB to Grid
	bool intersects = false;

	// Comparing two AABB (Grid Itself is an AABB)
	//...
	//...

	if(intersects)
	{
		writeIndex = atomicAdd(&gIndicesIndex, 0);
		gObjectIndices[writeIndex] = globalId;
	}
}

// Reconstruct SVO
// Creates SVO tree top down manner
// Implementation is opposite of parallel reduction
// Call Logic "per svo node (varying)"
__global__ void ReconstructSVO()
{

}


// Voxelize
// Do Voxelization of a mesh
// Its better to use OGL here since we render triangles to a array




#endif //__GIKERNELS_H__