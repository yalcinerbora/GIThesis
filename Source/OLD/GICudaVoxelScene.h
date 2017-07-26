/**

Voxel Representation of the Scene

*/

#ifndef __GICUDAVOXELSCENE_H__
#define __GICUDAVOXELSCENE_H__

#include "GICudaAllocator.h"
#include "StructuredBuffer.h"
#include "VoxelDebugVAO.h"
#include "VoxelCacheData.h"

class IEVector3;

class GICudaVoxelScene
{
	private:
		GICudaAllocator		allocator;

	protected:

	public:
		// Constructors & Destructor
							GICudaVoxelScene(const IEVector3& intialCenterPos, float span, unsigned int dim);
							GICudaVoxelScene(GICudaVoxelScene&&);
							GICudaVoxelScene(const GICudaVoxelScene&) = delete;
		GICudaVoxelScene&	operator=(const GICudaVoxelScene&) = delete;
							~GICudaVoxelScene();

		// Determines and Allocates the initial Page Size for the first frame
		void				LinkOGL(GLuint aabbBuffer,
									GLuint transformBuffer,
									GLuint jointTransformBuffer,
									GLuint transformIDBuffer,
									GLuint infoBufferID,
									GLuint voxelCacheNormPos,
									GLuint voxelCacheIds,
									GLuint voxelCacheRender,
									GLuint weightBuffer,
									uint32_t objCount,
									uint32_t voxelCount);
		void				AllocateWRTLinkedData(float coverageRatio);
		void				Reset();

		// Voxelize this current frame
		// Deletes moved voxels from grid
		// Adds newly entered voxels from the cache
		// Repositions existing voxels which is already in the grid
		// Reconstructs SVO tree
		IEVector3			VoxelUpdate(double& ioTiming,
										double& updateTiming,
										const IEVector3& playerPos,
										float cascadeMultiplier);

		// OGL Buffer Mapping
		void				MapGLPointers();
		void				UnmapGLPointers();

		// Debug Related Functions
		// Access for voxel data for rendering voxels
		uint64_t			AllocatorMemoryUsage() const;
		uint32_t			VoxelCountInPage();
		double				VoxDataToGL(CVoxelNormPos* dVAONormPosData,
										uchar4* dVAOColorData,

										CVoxelGrid& voxGridData,
										uint32_t& voxCount,
										uint32_t maxVoxelCount);
		GICudaAllocator*	Allocator();

};
#endif //__GICUDAVOXELSCENE_H__