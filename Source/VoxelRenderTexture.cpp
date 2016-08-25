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
				   GL_RG32UI,
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
	glBindImageTexture(index, texId, 0, GL_TRUE, 0, access, GL_RG32UI);
}

void VoxelRenderTexture::Clear()
{
	GLuint ff[2] = { 0xFFFFFFFF, 0xFFFFFFFF};
	glClearTexImage(texId, 0, GL_RG_INTEGER, GL_UNSIGNED_INT, &ff);
}