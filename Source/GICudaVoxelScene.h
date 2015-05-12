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

	protected:

	public:

		// Determines and Allocates the initial Page Size for the first frame
		void				AllocateInitialPages();

		// Voxelize this current frame
		// Deletes moved voxels from grid
		// Adds newly entered voxels from the cache
		// Repositions existing voxels which is already in the grid
		// Reconstructs SVO tree
		void				Voxelize(const IEVector3& playerPos);


		// Debug Related Functions
		void				VoxelDataForRendering(GLuint* mappedVoxelPages);

};

#endif //__GICUDAVOXELSCENE_H__