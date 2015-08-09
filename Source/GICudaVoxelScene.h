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
struct VoxelData
{
	uint32_t vox[4];
};

struct VoxelRenderData
{
	uint32_t color;
};
#pragma pack(pop)

class GICudaVoxelScene
{
	private:
		GICudaAllocator						allocator;

		StructuredBuffer<VoxelData>			vaoData;
		StructuredBuffer<VoxelRenderData>	vaoRenderData;
		VoxelDebugVAO						voxelVAO;


	protected:

	public:
		// Constructors & Destructor
							GICudaVoxelScene(const CVoxelGrid& gridSetup);
							GICudaVoxelScene(const GICudaVoxelScene&) = delete;
		GICudaVoxelScene&	operator=(const GICudaVoxelScene&) = delete;
							~GICudaVoxelScene();


		// Determines and Allocates the initial Page Size for the first frame
		void				LinkOGL(GLuint aabbBuffer,
									GLuint transformBufferID,
									GLuint relativeTransformBufferID,
									GLuint infoBufferID,
									GLuint voxelCache,
									GLuint voxelCacheRender,
									uint32_t objCount,
									uint32_t voxelCount);
		void				LinkSceneTextures(const std::vector<GLuint>& shadowMaps);
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
		void				Voxelize(const IEVector3& playerPos);

		// Debug Related Functions
		// Access for voxel data for rendering voxels
		VoxelDebugVAO&		VoxelDataForRendering(uint32_t& voxCount);


};
#endif //__GICUDAVOXELSCENE_H__