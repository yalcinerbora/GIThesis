#include "VoxelRenderTexture.h"
#include "GLHeader.h"

VoxelRenderTexture::VoxelRenderTexture()
	: texId(0)
{
	// To the GL
	glGenTextures(1, &texId);
	glBindTexture(GL_TEXTURE_3D, texId);
	glTexStorage3D(GL_TEXTURE_3D,
				   1,
				   GL_RGBA16UI,
				   VOXEL_GRID_SIZE,
				   VOXEL_GRID_SIZE,
				   VOXEL_GRID_SIZE);
}

VoxelRenderTexture::~VoxelRenderTexture()
{
	glDeleteTextures(1, &texId);
}

void VoxelRenderTexture::BindAsImage(uint32_t index, GLenum access)
{
	glBindImageTexture(index, texId, 0, GL_TRUE, 0, access, GL_RGBA16UI);
}

void VoxelRenderTexture::Clear()
{
	GLushort ff[4] = { 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF };
	glClearTexImage(texId, 0, GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, &ff);
}