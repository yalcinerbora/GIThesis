/**

Voxel Representation of the Scene

*/

#ifndef __GICUDAVOXELSCENE_H__
#define __GICUDAVOXELSCENE_H__

#include "GICudaAllocator.h"
#include "StructuredBuffer.h"
#include "VoxelDebugVAO.h"

class IEVector3;

#pragma pack(push, 1)
struct VoxelNormPos
{
	uint32_t vNormPos[2];
};

struct VoxelIds
{
	uint32_t vIds[2];
};

struct VoxelRenderData
{
	uint32_t color;				// Color
	// That is it for transform dynamic options
};

struct VoxelRenderSkelMorphData
{
	uchar4			color;		// Color

	// Transform Related Data
	// For Skeletal mesh these shows index of the transforms and weights
	// For Morph target this shows the neigbouring vertices and their morph related index
	uchar4			weightIndex;
	uchar4			weight;
};
#pragma pack(pop)

class GICudaVoxelScene
{
	private:
		GICudaAllocator						allocator;

		StructuredBuffer<VoxelNormPos>		vaoNormPosData;
		StructuredBuffer<uchar4>			vaoColorData;
		cudaGraphicsResource_t				vaoNormPosResource;
		cudaGraphicsResource_t				vaoRenderResource;
		cudaStream_t						stream;

	protected:

	public:
		// Constructors & Destructor
							GICudaVoxelScene(const IEVector3& intialCenterPos, float span, unsigned int dim);
							GICudaVoxelScene(const GICudaVoxelScene&) = delete;
		GICudaVoxelScene&	operator=(const GICudaVoxelScene&) = delete;
							~GICudaVoxelScene();

		static void			InitCuda();

		// Determines and Allocates the initial Page Size for the first frame
		void				LinkOGL(GLuint aabbBuffer,
									GLuint transformBufferID,
									GLuint infoBufferID,
									GLuint voxelCacheNormPos,
									GLuint voxelCacheIds,
									GLuint voxelCacheRender,
									uint32_t objCount,
									uint32_t voxelCount);
		void				AllocateWRTLinkedData(float coverageRatio);
		void				Reset();

		// Voxelize this current frame
		// Deletes moved voxels from grid
		// Adds newly entered voxels from the cache
		// Repositions existing voxels which is already in the grid
		// Reconstructs SVO tree
		void				VoxelUpdate(double& ioTiming,
										double& updateTiming,
										const IEVector3& playerPos);

		// Debug Related Functions
		// Access for voxel data for rendering voxels
		uint64_t			AllocatorMemoryUsage() const;
		uint32_t			VoxelCountInPage();
		VoxelDebugVAO		VoxelDataForRendering(CVoxelGrid&, double& timing, uint32_t& voxCount, bool isOuterCascade);
		GICudaAllocator*	Allocator();
};
#endif //__GICUDAVOXELSCENE_H__