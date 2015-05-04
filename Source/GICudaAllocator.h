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
#include "GLHeaderLite.h"

#define GI_DELETED_VOXEL	0xFFFFFFFF
#define GI_STATIC_GEOMETRY	0

#define GI_PAGE_SIZE 65536
#define GI_THREAD_PER_BLOCK 512
#define GI_BLOCK_PER_PAGE GI_PAGE_SIZE / GI_THREAD_PER_BLOCK

static_assert(GI_PAGE_SIZE % GI_THREAD_PER_BLOCK == 0, "Page size must be divisible by thread per block");


struct CVoxelData
{
	CVoxelPacked*		dGridVoxels;
	CVoxelRender*		dVoxelsRenderData;
	unsigned int*		dEmptyPos;
	unsigned int		dEmptyElementIndex;
};

class GICudaAllocator
{
	
	private:
		// Grid Data
		std::vector<CVoxelData> hVoxelPages;
		CVoxelData*				dVoxelPages;
		CVoxelGrid				dVoxelGridInfo;

		// Object Related Data
		CObjectTransform*		dRelativeTransform;	// Transform matrices relative to the prev frame (world -> world)
		CObjectTransform*		dTransform;			// Transform matrices from object space (object -> world)
		CObjectAABB*			dObjectAABB;		// Object Space Axis Aligned Bounding Box for each object
		CObjectVoxelInfo*		dObjectInfo;		// Voxel Count of the object

		CVoxelPacked**			dObjCache;			
		CVoxelRender**			dObjRenderCache;	

		size_t					objectCount;
		size_t					pageCount;

	protected:
	public:
		// Constructors & Destructor


		// Utility
		uint32_t	CreateVoxelCache(GLuint& createdBuffer,
									 size_t voxelCount,
									 const CObjectAABB& aabb,
									 const CObjectVoxelInfo& info);
		bool		DeleteVoxelCache(uint32_t cacheId);
		void		LinkTransformBufferFromOGL(GLuint bufferId);
		void		LinkRelativeTransformBufferFromOGL(GLuint bufferId);

		// Voxel Page Related
		void		AddVoxelPage(size_t count);
		void		ShrinkVoxelPages(size_t pageCount);




};
#endif //__GICUDAALLOCATOR_H_