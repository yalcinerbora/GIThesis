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
#include "CudaVector.cuh"
#include "GLHeader.h"
#include "CVoxelPage.h"
#include <cudaGL.h>
#include "IEUtility/IEVector3.h"

struct CVoxelPageData
{
	CudaVector<CVoxelNormPos>	dVoxelPageNormPos;
	CudaVector<CVoxelIds>		dVoxelPageIds;
	CudaVector<unsigned int>	dEmptySegmentList;
	CudaVector<char>			dIsSegmentOccupied;

	CVoxelPageData(size_t sizeOfPage, size_t sizeOfHelper)
		: dVoxelPageNormPos(sizeOfPage)
		, dVoxelPageIds(sizeOfPage)
		, dEmptySegmentList(sizeOfHelper)
		, dIsSegmentOccupied(sizeOfHelper)
	{}
};

class GICudaAllocator
{
	
	private:
		// Grid Data
		std::vector<CVoxelPage>					hVoxelPages;
		CudaVector<CVoxelPage>					dVoxelPages;
		std::vector<CVoxelPageData>				hPageData;

		CVoxelGrid								hVoxelGridInfo;
		CudaVector<CVoxelGrid>					dVoxelGridInfo;

		// Helper Data (That is populated by system)
		// Object Segment Related
		std::vector<CudaVector<unsigned int>>	dSegmentObjecId;
		std::vector<CudaVector<ushort2>>		dSegmentAllocLoc;

		// Per Object
		std::vector<CudaVector<unsigned int>>	dVoxelStrides;
		std::vector<CudaVector<unsigned int>>	dObjectAllocationIndexLookup;
		std::vector<CudaVector<char>>			dWriteSignals;

		// Array of Device Pointers
		CudaVector<unsigned int*>				dObjectAllocationIndexLookup2D;
		CudaVector<unsigned int*>				dObjectVoxStrides2D;
		CudaVector<ushort2*>					dSegmentAllocLoc2D;
		//------

		// Object Related Data (Comes from OGL)
		// Kernel call ready aligned pointer(s)		
		CudaVector<CObjectTransform*>			dTransforms;			// Transform matrices from object space (object -> world)
		CudaVector<CObjectAABB*>				dObjectAABB;			// Object Space Axis Aligned Bounding Box for each object
		CudaVector<CObjectVoxelInfo*>			dObjectInfo;			// Voxel Count of the object
		
		CudaVector<CVoxelPacked*>				dObjCache;
		CudaVector<CVoxelRender*>				dObjRenderCache;

		std::vector<CObjectTransform*>			hTransforms;			
		std::vector<CObjectAABB*>				hObjectAABB;			
		std::vector<CObjectVoxelInfo*>			hObjectInfo;		

		std::vector<CVoxelPacked*>				hObjCache;
		std::vector<CVoxelRender*>				hObjRenderCache;

		// G Buffer Related Data
		cudaTextureObject_t						depthBuffer;
		cudaTextureObject_t						normalBuffer;
		cudaSurfaceObject_t						lightIntensityBuffer;

		// Scene Light Related Data
		cudaTextureObject_t						shadowMaps;

		// Interop Data
		std::vector<cudaGraphicsResource_t>		transformLinks;
		std::vector<cudaGraphicsResource_t>		aabbLinks;
		std::vector<cudaGraphicsResource_t>		objectInfoLinks;

		std::vector<cudaGraphicsResource_t>		cacheLinks;
		std::vector<cudaGraphicsResource_t>		cacheRenderLinks;

		// Per Scene Interop Data
		cudaGraphicsResource_t					sceneShadowMapLink;	
		cudaGraphicsResource_t					depthBuffLink;
		cudaGraphicsResource_t					normalBuffLink;
		cudaGraphicsResource_t					lightIntensityLink;

		// Size Data
		std::vector<size_t>						voxelCounts;
		std::vector<size_t>						objectCounts;
		size_t									totalObjectCount;

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
												  GLuint infoBufferID,
												  GLuint voxelCache,
												  GLuint voxelCacheRender,
												  uint32_t objCount,
												  uint32_t voxelCount);
		void					LinkSceneShadowMapArray(GLuint shadowMapArray);
		void					LinkSceneGBuffers(GLuint depthTex,
												  GLuint normalTex,
												  GLuint lightIntensityTex);
		void					UnLinkGBuffers();

		// Resetting Scene related data (called when scene changes)
		void					ResetSceneData();
		void					Reserve(uint32_t pageAmount);

		void					SendNewVoxPosToDevice();

		// Mapping OGL (mapped unmapped each frame)
		void					SetupDevicePointers();
		void					ClearDevicePointers();

		uint32_t				NumObjectBatches() const;
		uint32_t				NumObjects(uint32_t batchIndex) const;
		uint32_t				NumObjectSegments(uint32_t batchIndex) const;
		uint32_t				NumVoxels(uint32_t batchIndex) const;
		uint32_t				NumPages() const;

		CVoxelGrid*				GetVoxelGridDevice();
		CVoxelGrid				GetVoxelGridHost();
		IEVector3				GetNewVoxelPos(const IEVector3& playerPos);

		// Mapped OGL Pointers		
		CObjectTransform**		GetTransformsDevice();
		CObjectAABB**			GetObjectAABBDevice();
		CObjectVoxelInfo**		GetObjectInfoDevice();

		CVoxelPacked**			GetObjCacheDevice();
		CVoxelRender**			GetObjRenderCacheDevice();

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
		unsigned int**			GetObjectVoxStrides2D();
		ushort2**				GetSegmentAllocLoc2D();
};
#endif //__GICUDAALLOCATOR_H_