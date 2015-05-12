/**

	 VOXEL Render TExture

*/


#ifndef __VOXELRENDERTEXTURE_H__
#define __VOXELRENDERTEXTURE_H__

#include "GLHeaderLite.h"
#include <cstdint>

#define VOXEL_SIZE 128

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