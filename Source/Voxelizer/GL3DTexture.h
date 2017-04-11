/**

3D Texture for Voxel Operations

*/


#ifndef __GL3DTEXTURE_H__
#define __GL3DTEXTURE_H__

#include "GLHeader.h"
#include <cstdint>

enum class TextureDataType
{
	UINT_1 = GL_R32UI,
	FLOAT_4 = GL_RGBA32F,
};

#define VOX_3D_TEX_SIZE 256
//#define VOX_3D_TEX_SIZE 512

class GL3DTexture
{
	private:
		GLuint						texId;
		TextureDataType				type;

	protected:

	public:
									GL3DTexture(TextureDataType,
												GLsizei dimX = VOX_3D_TEX_SIZE,
												GLsizei dimY = VOX_3D_TEX_SIZE,
												GLsizei dimZ = VOX_3D_TEX_SIZE);
									GL3DTexture(const GL3DTexture&) = delete;
		const GL3DTexture&			operator=(const GL3DTexture&) = delete;
									~GL3DTexture();

		GLuint						TexId();
		void						BindAsImage(GLuint size, GLenum access);
		void						Clear();
};
#endif //__GL3DTEXTURE_H__