/**

	 VOXEL Render TExture

*/


#ifndef __VOXELRENDERTEXTURE_H__
#define __VOXELRENDERTEXTURE_H__

#include "GLHeaderLite.h"
#include <cstdint>

// My kepler card has 2GB of ram (with 4GB virtual memory)
// 512 3D texture has some issues on the card (some deadlocks etc.)
// prob there is some driver/card bug somwhere
// 416 seems to be working tho so its k
#define VOXEL_GRID_SIZE 512

class VoxelRenderTexture
{
	private:
		GLuint						texId;

	protected:

	public:
									VoxelRenderTexture();
									VoxelRenderTexture(const VoxelRenderTexture&) = delete;
		const VoxelRenderTexture&	operator=(const VoxelRenderTexture&) = delete;
									~VoxelRenderTexture();

		void						BindAsImage(uint32_t index,  GLenum access);
		void						Clear();
};
#endif //__VOXELRENDERTEXTURE_H__