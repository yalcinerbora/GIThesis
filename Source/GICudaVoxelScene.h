/**

Voxel Representation of the Scene

*/

#ifndef __GICUDAVOXELSCENE_H__
#define __GICUDAVOXELSCENE_H__

#include "GICudaAllocator.h"

class IEVector3;

class GICudaVoxelScene
{
	private:
		GICudaAllocator		allocator;

		CVoxelGrid*			dVoxGrid;
		CVoxelGrid			hVoxGrid;

	protected:

	public:
		// Constructors & Destructor
							GICudaVoxelScene();
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
									size_t objCount);
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
		GLuint				VoxelDataForRendering();


};

#endif //__GICUDAVOXELSCENE_H__