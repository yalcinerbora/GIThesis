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
	uint32_t color;

	// For Moprh Targets
	// uint3 parentVertexIndex
	// float3 vertexWeights

	// For Skeleton Anim
	// bone index (at most 4)
	// bone weights (at most 4)

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
		void				LinkSceneTextures(GLuint shadowMapArray);
		void				LinkDeferredRendererBuffers(GLuint depthBuffer,
														GLuint normalGBuff,
														GLuint lightIntensityTex);
		void				UnLinkDeferredRendererBuffers();
		void				AllocateInitialPages(uint32_t approxVoxCount);
		void				Reset();

		// Voxelize this current frame
		// Deletes moved voxels from grid
		// Adds newly entered voxels from the cache
		// Repositions existing voxels which is already in the grid
		// Reconstructs SVO tree
		void				VoxelUpdate(double& ioTiming,
										double& updateTiming,
										double& svoReconsTiming,
										const IEVector3& playerPos);

		// Debug Related Functions
		// Access for voxel data for rendering voxels
		uint64_t			AllocatorMemoryUsage() const;
		uint32_t			VoxelCountInPage();
		VoxelDebugVAO		VoxelDataForRendering(CVoxelGrid&, double& timing, uint32_t voxCount);
};
#endif //__GICUDAVOXELSCENE_H__