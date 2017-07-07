/**

Memory Allocation

*/

#ifndef __GICUDAALLOCATOR_H__
#define __GICUDAALLOCATOR_H__

#include <cuda.h>
#include <stdint.h>
#include <vector>
#include "CVoxelTypes.h"
#include "COpenGLCommon.cuh"
#include "CudaVector.cuh"
#include "GLHeader.h"
#include "GIVoxelPages.h"
#include <cudaGL.h>
#include "IEUtility/IEVector3.h"



struct CVoxelPageData
{
	CudaVector<CVoxelPos>			dVoxelPagePos;
	CudaVector<CVoxelNorm>			dVoxelPageNorm;
	CudaVector<CVoxelOccupancy>		dVoxelOccupancy;
	CudaVector<unsigned char>		dEmptySegmentList;
	CudaVector<SegmentObjData>		dIsSegmentOccupied;

	CVoxelPageData() {};
	CVoxelPageData(size_t sizeOfPage, size_t sizeOfHelper)
		: dVoxelPagePos(sizeOfPage)
		, dVoxelPageNorm(sizeOfPage)
		, dVoxelOccupancy(sizeOfPage)
		, dEmptySegmentList(sizeOfHelper)
		, dIsSegmentOccupied(sizeOfHelper)
	{}
};

class GICudaAllocator
{
	
	private:
		static const unsigned int				SVOTextureSize;
		static const unsigned int				SVOTextureDepth;

		// Grid Data
		std::vector<CVoxelPage>					hVoxelPages;
		CudaVector<CVoxelPage>					dVoxelPages;
		std::vector<CVoxelPageData>				hPageData;
		size_t									reservedPageCount;

		CVoxelGrid								hVoxelGridInfo;
		CudaVector<CVoxelGrid>					dVoxelGridInfo;

		// Helper Data (That is populated by system)
		// Object Segment Related
		std::vector<CudaVector<SegmentObjData>>	dSegmentObjecId;
		std::vector<CudaVector<ushort2>>	 	dSegmentAllocLoc;

		// Per Object
		std::vector<CudaVector<unsigned int>>	dVoxelStrides;
		std::vector<CudaVector<unsigned int>>	dObjectAllocationIndexLookup;

		// Array of Device Pointers
		CudaVector<unsigned int*>			 	dObjectAllocationIndexLookup2D;
		CudaVector<unsigned int*>			 	dObjectVoxStrides2D;
		CudaVector<ushort2*>				 	dSegmentAllocLoc2D;
		//------

		// Object Related Data (Comes from OGL)
		// Kernel call ready aligned pointer(s)		
		CudaVector<CObjectTransform*>			dTransforms;			// Transform matrices from object space (object -> world)
		CudaVector<uint32_t*>					dTransformIds;
		CudaVector<CObjectAABB*>				dObjectAABB;			// Object Space Axis Aligned Bounding Box for each object
		CudaVector<CObjectVoxelInfo*>			dObjectInfo;			// Voxel Count of the object
		CudaVector<CObjectTransform*>			dJointTransform;		// Joint Transforms

		CudaVector<CVoxelNormPos*>				dObjNormPosCache;
		//CudaVector<CVoxelIds*>				dObjIdsCache;
		CudaVector<CVoxelColor*>				dObjRenderCache;
		CudaVector<CVoxelWeight*>				dObjWeight;

		std::vector<CObjectTransform*>			hTransforms;
		std::vector<CObjectAABB*>				hObjectAABB;
		std::vector<uint32_t*>					hTransformIds;
		std::vector<CObjectVoxelInfo*>			hObjectInfo;
		std::vector<CObjectTransform*>			hJointTransform;

		std::vector<CVoxelNormPos*>				hObjNormPosCache;
		//std::vector<CVoxelIds*>				hObjIdsCache;
		std::vector<CVoxelColor*>				hObjRenderCache;
		std::vector<CVoxelWeight*>				hObjWeight;

		// Interop Data
		std::vector<cudaGraphicsResource_t>		transformLinks;
		std::vector<cudaGraphicsResource_t>		transformIdLinks;
		std::vector<cudaGraphicsResource_t>		aabbLinks;
		std::vector<cudaGraphicsResource_t>		objectInfoLinks;
		std::vector<cudaGraphicsResource_t>		jointTransformLinks;

		std::vector<cudaGraphicsResource_t>		cacheNormPosLinks;
		//std::vector<cudaGraphicsResource_t>	cacheIdsLinks;
		std::vector<cudaGraphicsResource_t>		cacheRenderLinks;
		std::vector<cudaGraphicsResource_t>		objWeightLinks;

		// Size Data
		std::vector<size_t>						voxelCounts;
		std::vector<size_t>						objectCounts;
		size_t									totalSegmentCount;
		size_t									totalObjectCount;

		bool									pointersSet;

		// Voxel Page System memory Mangement
		void					AddVoxelPages(size_t count);
		void					RemoveVoxelPages(size_t count);

	protected:
	public:
		// Constructors & Destructor
								GICudaAllocator(const CVoxelGrid& gridInfo);
								~GICudaAllocator() = default;

		// Linking and Unlinking Voxel Cache Data (from OGL)
		void					LinkOGLVoxelCache(GLuint aabbBuffer,
												  GLuint transformBuffer,
												  GLuint jointTransformBuffer,
												  GLuint transformIDBuffer,
												  GLuint infoBuffer,
												  GLuint voxelNormPosBuffer,
												  GLuint voxelIdsBuffer,
												  GLuint voxelCacheRender,
												  GLuint voxelWeightBuffer,
												  uint32_t objCount,
												  uint32_t voxelCount);

		// Resetting Scene related data (called when scene changes)
		void					ResetSceneData();
		void					ReserveForSegments(float coverageRatio);

		void					SendNewVoxPosToDevice();

		// Mapping OGL (mapped unmapped each frame)
		void					SetupDevicePointers();
		void					ClearDevicePointers();

		uint32_t				NumObjectBatches() const;
		uint32_t				NumObjects(uint32_t batchIndex) const;
		uint32_t				NumObjectSegments(uint32_t batchIndex) const;
		uint32_t				NumVoxels(uint32_t batchIndex) const;
		uint32_t				NumPages() const;
		uint32_t				NumSegments() const;

		CVoxelGrid*				GetVoxelGridDevice();
		const CVoxelGrid&		GetVoxelGridHost() const;
		IEVector3				GetNewVoxelPos(const IEVector3& playerPos,
											   float cascadeMultiplier);

		// Memory Usage Func
		uint64_t				SystemTotalMemoryUsage() const;

		// Mapped OGL Pointers		
		CObjectTransform**		GetTransformsDevice();
		uint32_t**				GetTransformIDDevice();
		CObjectAABB**			GetObjectAABBDevice();
		CObjectVoxelInfo**		GetObjectInfoDevice();
		CObjectTransform**		GetJointTransformDevice();

		CVoxelNormPos**			GetObjCacheNormPosDevice();
		//CVoxelIds**				GetObjCacheIdsDevice();
		CVoxelColor**			GetObjRenderCacheDevice();
		CVoxelWeight**			GetObjWeightDevice();

		CObjectTransform*		GetTransformsDevice(uint32_t index);
		uint32_t*				GetTransformIDDevice(uint32_t index);
		CObjectAABB*			GetObjectAABBDevice(uint32_t index);
		CObjectVoxelInfo*		GetObjectInfoDevice(uint32_t index);

		CVoxelNormPos*			GetObjCacheNormPosDevice(uint32_t index);
		//CVoxelIds*			GetObjCacheIdsDevice(uint32_t index);
		CVoxelColor*			GetObjRenderCacheDevice(uint32_t index);

		// Pages
		CVoxelPage*				GetVoxelPagesDevice();

		// Helper Data (That is populated by system)
		// Object Segment Related
		SegmentObjData*			GetSegmentObjectID(uint32_t index);
		ushort2*				GetSegmentAllocLoc(uint32_t index);

		unsigned int*			GetVoxelStrides(uint32_t index);
		unsigned int*			GetObjectAllocationIndexLookup(uint32_t index);

		unsigned int**			GetObjectAllocationIndexLookup2D();
		unsigned int**			GetObjectVoxStrides2D();
		ushort2**				GetSegmentAllocLoc2D();

		bool					IsGLMapped();
};
#endif //__GICUDAALLOCATOR_H_