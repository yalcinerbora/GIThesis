#include "GL3DTexture.h"


GL3DTexture::GL3DTexture(TextureDataType type,
						 GLsizei dimX,
						 GLsizei dimY,
						 GLsizei dimZ)
	: type(type)
{
	glGenTextures(1, &texId);
	glBindTexture(GL_TEXTURE_3D, texId);
	glTexStorage3D(GL_TEXTURE_3D, 1,
				   static_cast<GLenum>(type),
				   dimX,
				   dimY,
				   dimZ);
}

GL3DTexture::~GL3DTexture()
{
	glDeleteTextures(1, &texId);
}

GLuint GL3DTexture::TexId()
{
	return texId;
}

void GL3DTexture::BindAsImage(GLuint index, GLenum access)
{
	glBindImageTexture(index, texId, 0, GL_TRUE, 0, access, static_cast<GLenum>(type));
}

void GL3DTexture::Clear()
{
	GLuint dataInt = 0x00000000;
	float dataFloat[4] = {0.0f, 0.0f, 0.0f, 0.0f};

	GLenum format, dataType;
	void* data = nullptr;
	if(type == TextureDataType::UINT_1)
	{
		format = GL_RED_INTEGER;
		dataType = GL_UNSIGNED_INT;
		data = &dataInt;
	}
	else if(type == TextureDataType::FLOAT_4)
	{
		format = GL_RGBA;
		dataType = GL_FLOAT;
		data = &dataFloat;
	}
	glClearTexImage(texId, 0, format, dataType, data);
}