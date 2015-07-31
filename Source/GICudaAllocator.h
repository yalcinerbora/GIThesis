/**

Memory Allocation

*/

#ifndef __GICUDAALLOCATOR_H__
#define __GICUDAALLOCATOR_H__

#include <cuda.h>
#include <stdint.h>
#include <vector>
#include "CVoxel.cuh"
#include "COpenGLCommon.cuh"
#include <thrust/device_vector.h>
#include "GLHeader.h"
#include "CVoxelPage.h"
#include <cudaGL.h>

class GICudaAllocator
{
	
	private:
		// Grid Data
		std::vector<CVoxelPage>							hVoxelPages;
		thrust::device_vector<CVoxelPage>				dVoxelPages;
		CVoxelGrid										dVoxelGridInfo;

		// Object Related Data (Comes from OGL)
		// Kernel call ready aligned pointer(s)
		thrust::device_vector<CObjectTransform*>		dRelativeTransforms;	// Transform matrices relative to the prev frame (world -> world)
		thrust::device_vector<CObjectTransform*>		dTransforms;			// Transform matrices from object space (object -> world)
		thrust::device_vector<CObjectAABB*>				dObjectAABB;			// Object Space Axis Aligned Bounding Box for each object
		thrust::device_vector<CObjectVoxelInfo*>		dObjectInfo;			// Voxel Count of the object

		thrust::device_vector<CVoxelPacked*>			dObjCache;
		thrust::device_vector<CVoxelRender*>			dObjRenderCache;
		
		// Interop Data
		std::vector<cudaGraphicsResource*>				rTransformLinks;
		std::vector<cudaGraphicsResource*>				transformLinks;
		std::vector<cudaGraphicsResource*>				aabbLinks;
		std::vector<cudaGraphicsResource*>				objectInfoLinks;

		std::vector<cudaGraphicsResource*>				cacheLinks;
		std::vector<cudaGraphicsResource*>				cacheRenderLinks;

		// Size Data
		std::vector<size_t>								objectCount;
		size_t											totalObjectCount;
		size_t											pageCount;
		
		//
		void					SetupPointersDevicePointers();
		void					ClearDevicePointers();

		//
		void					AddVoxelPage(size_t count);
		void					ShrinkVoxelPages(size_t pageCount);

	protected:
	public:
		// Constructors & Destructor
								GICudaAllocator();
								~GICudaAllocator() = default;

		// Linking and Unlinking Voxel Cache Data (from OGL)
		uint32_t				LinkOGLVoxelCache(GLuint batchAABBBuffer,
												  GLuint batchTransformBuffer,
												  GLuint relativeTransformBuffer,
												  GLuint infoBuffer,
												  GLuint voxelBuffer,
												  GLuint voxelRenderBuffer);

		void					UnlinkOGLVoxelCache(uint32_t index);
		
		
};
#endif //__GICUDAALLOCATOR_H_