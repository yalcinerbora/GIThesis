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
	thrust::device_vector<char>			dIsSegmentOccupied;
};

class GICudaAllocator
{
	
	private:
		// Grid Data
		thrust::host_vector<CVoxelPage>					hVoxelPages;
		thrust::device_vector<CVoxelPage>				dVoxelPages;
		std::vector<CVoxelPageData>						hPageData;

		CVoxelGrid										hVoxelGridInfo;
		thrust::device_vector<CVoxelGrid>				dVoxelGridInfo;

		// Helper Data (That is populated by system)
		// Object Segment Related
		std::vector<thrust::device_vector<unsigned int>>	dSegmentObjecId;
		std::vector<thrust::device_vector<ushort2>>			dSegmentAllocLoc;

		// Per Object
		std::vector<thrust::device_vector<unsigned int>>	dVoxelStrides;
		std::vector<thrust::device_vector<unsigned int>>	dObjectAllocationIndexLookup;
		std::vector<thrust::device_vector<char>>			dWriteSignals;

		// Array of Device Pointers
		thrust::device_vector<unsigned int*>				dObjectAllocationIndexLookup2D;
		thrust::device_vector<ushort2*>						dSegmentAllocLoc2D;
		//------

		// Object Related Data (Comes from OGL)
		// Kernel call ready aligned pointer(s)
		thrust::device_vector<CObjectTransform*>		dRelativeTransforms;	// Transform matrices relative to the prev frame (world -> world)
		thrust::device_vector<CObjectTransform*>		dTransforms;			// Transform matrices from object space (object -> world)
		thrust::device_vector<CObjectAABB*>				dObjectAABB;			// Object Space Axis Aligned Bounding Box for each object
		thrust::device_vector<CObjectVoxelInfo*>		dObjectInfo;			// Voxel Count of the object

		thrust::device_vector<CVoxelPacked*>			dObjCache;
		thrust::device_vector<CVoxelRender*>			dObjRenderCache;

		thrust::host_vector<CObjectTransform*>			hRelativeTransforms;	
		thrust::host_vector<CObjectTransform*>			hTransforms;			
		thrust::host_vector<CObjectAABB*>				hObjectAABB;			
		thrust::host_vector<CObjectVoxelInfo*>			hObjectInfo;		
								
		thrust::host_vector<CVoxelPacked*>				hObjCache;
		thrust::host_vector<CVoxelRender*>				hObjRenderCache;

		// G Buffer Related Data
		cudaTextureObject_t								depthBuffer;
		cudaTextureObject_t								normalBuffer;
		cudaSurfaceObject_t								lightIntensityBuffer;

		// Scene Light Related Data
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
		std::vector<size_t>								voxelCounts;
		std::vector<size_t>								objectCounts;
		size_t											totalObjectCount;

		//
		void					AddVoxelPage(size_t count);
		//void					ShrinkVoxelPages(size_t pageCount);

	protected:
	public:
		// Constructors & Destructor
								GICudaAllocator(const CVoxelGrid& gridInfo);
								~GICudaAllocator() = default;

		// Linking and Unlinking Voxel Cache Data (from OGL)
		void					LinkOGLVoxelCache(GLuint aabbBuffer,
												  GLuint transformBufferID,
												  GLuint relativeTransformBufferID,
												  GLuint infoBufferID,
												  GLuint voxelCache,
												  GLuint voxelCacheRender,
												  uint32_t objCount,
												  uint32_t voxelCount);
		void					LinkSceneShadowMapArray(const std::vector<GLuint>& shadowMaps);
		void					LinkSceneGBuffers(GLuint depthTex,
												  GLuint normalTex,
												  GLuint lightIntensityTex);
		void					UnLinkGBuffers();

		// Resetting Scene related data (called when scene changes)
		void					ResetSceneData();
		void					Reserve(uint32_t pageAmount);

		// Mapping OGL (mapped unmapped each frame)
		void					SetupDevicePointers();
		void					ClearDevicePointers();

		uint32_t				NumObjectBatches() const;
		uint32_t				NumObjects(uint32_t batchIndex) const;
		uint32_t				NumObjectSegments(uint32_t batchIndex) const;
		uint32_t				NumVoxels(uint32_t batchIndex) const;
		uint32_t				NumPages() const;

		CVoxelGrid*				GetVoxelGridDevice();

		// Mapped OGL Pointers
		CObjectTransform**		GetRelativeTransformsDevice();
		CObjectTransform**		GetTransformsDevice();
		CObjectAABB**			GetObjectAABBDevice();
		CObjectVoxelInfo**		GetObjectInfoDevice();

		CVoxelPacked**			GetObjCacheDevice();
		CVoxelRender**			GetObjRenderCacheDevice();

		CObjectTransform*		GetRelativeTransformsDevice(uint32_t index);
		CObjectTransform*		GetTransformsDevice(uint32_t index);
		CObjectAABB*			GetObjectAABBDevice(uint32_t index);
		CObjectVoxelInfo*		GetObjectInfoDevice(uint32_t index);

		CVoxelPacked*			GetObjCacheDevice(uint32_t index);
		CVoxelRender*			GetObjRenderCacheDevice(uint32_t index);

		// Pages
		CVoxelPage*				GetVoxelPagesDevice();

		// Helper Data (That is populated by system)
		// Object Segment Related
		unsigned int*			GetSegmentObjectID(uint32_t index);
		ushort2*				GetSegmentAllocLoc(uint32_t index);

		unsigned int*			GetVoxelStrides(uint32_t index);
		unsigned int*			GetObjectAllocationIndexLookup(uint32_t index);
		char*					GetWriteSignals(uint32_t index);

		unsigned int**			GetObjectAllocationIndexLookup2D();
		ushort2**				GetSegmentAllocLoc2D();
};
#endif //__GICUDAALLOCATOR_H_