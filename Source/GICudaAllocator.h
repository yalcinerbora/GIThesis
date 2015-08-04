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

struct CVoxelPageData
{
	thrust::device_vector<CVoxelPacked> dVoxelPage;
	thrust::device_vector<unsigned int> dEmptySegmentList;
};

class GICudaAllocator
{
	
	private:
		// Grid Data
		thrust::host_vector<CVoxelPage>					hVoxelPages;
		thrust::device_vector<CVoxelPage>				dVoxelPages;
		std::vector<CVoxelPageData>						hPageData;

		CVoxelGrid										hVoxelGridInfo;


		// Helper Data (That is populated by system)
		// Object Segment Related
		std::vector<thrust::device_vector<unsigned int*>>	dSegmentObjecId;
		std::vector<thrust::device_vector<ushort2*>>		dSegmentAllocLoc;

		// Per Object
		std::vector<thrust::device_vector<unsigned int*>>	dVoxelStrides;
		std::vector<thrust::device_vector<unsigned int*>>	dObjectAllocationIndexLookup;
		std::vector<thrust::device_vector<char*>>			dWriteSignals;
		//------


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