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
#include <cudaGL.h>

#define GI_DELETED_VOXEL	0xFFFFFFFF
#define GI_STATIC_GEOMETRY	0

#define GI_PAGE_SIZE 65536
#define GI_THREAD_PER_BLOCK 256
#define GI_BLOCK_PER_PAGE GI_PAGE_SIZE / GI_THREAD_PER_BLOCK
#define GI_SEGMENT_SIZE GI_BLOCK_PER_PAGE

static_assert(GI_PAGE_SIZE % GI_THREAD_PER_BLOCK == 0, "Page size must be divisible by thread per block");

struct CVoxelPage
{
	CVoxelPacked*		dGridVoxels;
	CVoxelRender**		dVoxelsRenderDataPtr;
	unsigned int*		dEmptySegmentPos;
	unsigned int		dEmptySegmentIndex;
};

struct CVoxelPageData
{
	thrust::device_vector<CVoxelPacked> dVoxelPage;
	thrust::device_vector<CVoxelRender> dVoxelPageRender;
	thrust::device_vector<unsigned int> dVoxelState;
	thrust::device_vector<unsigned int> dEmptySegmentPos;
};

class GICudaAllocator
{
	
	private:
		// Grid Data
		thrust::host_vector<CVoxelPage>					hVoxelPages;
		thrust::device_vector<CVoxelPage>				dVoxelPages;
		std::vector<CVoxelPageData>						hPageData;

		CVoxelGrid										dVoxelGridInfo;

		// Object Related Data (Comes from OGL)
		// Kernel call ready aligned pointer(s)
		thrust::device_vector<CObjectTransform*>		dRelativeTransforms;	// Transform matrices relative to the prev frame (world -> world)
		thrust::device_vector<CObjectTransform*>		dTransforms;			// Transform matrices from object space (object -> world)
		thrust::device_vector<CObjectAABB*>				dObjectAABB;			// Object Space Axis Aligned Bounding Box for each object
		thrust::device_vector<CObjectVoxelInfo*>		dObjectInfo;			// Voxel Count of the object

		thrust::device_vector<CVoxelPacked*>			dObjCache;
		thrust::device_vector<CVoxelRender*>			dObjRenderCache;
		
		cudaTextureObject_t								depthBuffer;
		cudaTextureObject_t								normalBuffer;
		cudaSurfaceObject_t								lightIntensityBuffer;
		std::vector<cudaTextureObject_t>				shadowMaps;

		// Interop Data
		std::vector<cudaGraphicsResource*>				rTransformLinks;
		std::vector<cudaGraphicsResource*>				transformLinks;
		std::vector<cudaGraphicsResource*>				aabbLinks;
		std::vector<cudaGraphicsResource*>				objectInfoLinks;

		std::vector<cudaGraphicsResource*>				cacheLinks;
		std::vector<cudaGraphicsResource*>				cacheRenderLinks;

		// Per Scene Interop Data
		std::vector<cudaGraphicsResource*>				sceneShadowMapLinks;
		cudaGraphicsResource*							depthBuffLink;
		cudaGraphicsResource*							normalBuffLink;
		cudaGraphicsResource*							lightIntensityLink;

		// Size Data
		std::vector<size_t>								objectCounts;
		size_t											totalObjectCount;
		size_t											pageCount;
		
		//
		void					SetupPointersDevicePointers();
		void					ClearDevicePointers();

		//
		void					AddVoxelPage(size_t count);
		//void					ShrinkVoxelPages(size_t pageCount);

	protected:
	public:
		// Constructors & Destructor
								GICudaAllocator();
								~GICudaAllocator() = default;

		// Linking and Unlinking Voxel Cache Data (from OGL)
		void					LinkOGLVoxelCache(GLuint aabbBuffer,
												  GLuint transformBufferID,
												  GLuint relativeTransformBufferID,
												  GLuint infoBufferID,
												  GLuint voxelCache,
												  GLuint voxelCacheRender,
												  size_t objCount);
		void					LinkSceneShadowMapArray(const std::vector<GLuint>& shadowMaps);
		void					LinkSceneGBuffers(GLuint depthTex,
												  GLuint normalTex,
												  GLuint lightIntensityTex);

		// Resetting Scene related data (called when scene changes)
		void					ResetSceneData();

		// Mapped OGL Pointers
		const CObjectTransform**		GetRelativeTransformsDevice();
		const CObjectTransform**		GetTransformsDevice();
		const CObjectAABB**				GetObjectAABBDevice();
		const CObjectVoxelInfo**		GetObjectInfoDevice();

		const CVoxelPacked**			GetObjCacheDevice();
		const CVoxelRender**			GetObjRenderCacheDevice();
		
		// Pages
		CVoxelPage*						GetVoxelPagesDevice();
		
};
#endif //__GICUDAALLOCATOR_H_