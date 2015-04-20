#include "Material.h"
#include "Globals.h"
#include <fstream>
#include "TGALoad.h"
#include "Macros.h"

Material::Material(ColorMaterial c)
	: texture(0)
	, sampler(0)
{
	// Load Texture
	// Texture is Targa
	TGAFILE tga;
	LoadTGAFile(&tga, c.colorFileName);

	// Has to be RGB uncompressed
	assert(tga.imageTypeCode == 2);

	// To the GL
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_2D, texture);
	glTexStorage2D(GL_TEXTURE_2D,
				   4,
				   (tga.bitCount == 24) ? GL_RGB8 : GL_RGBA8,
				   tga.imageWidth,
				   tga.imageHeight);

	// Do the Actual Loading
	glTexSubImage2D(GL_TEXTURE_2D,
					0,
					0,
					0,
					tga.imageWidth,
					tga.imageHeight,
					(tga.bitCount == 24) ? GL_RGB : GL_RGBA,
					GL_UNSIGNED_BYTE, 
					tga.imageData);

	glGenerateMipmap(GL_TEXTURE_2D);

	// Tex Parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 8);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_GREEN);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_BLUE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, GL_ALPHA);

	glGenSamplers(1, &sampler);
	glSamplerParameteri(sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glSamplerParameterf(sampler, GL_TEXTURE_MAX_ANISOTROPY_EXT, 8.0f);

	free(tga.imageData);
}

Material::Material(Material&& other)
	: texture(other.texture)
	, sampler(other.sampler)
{
	other.texture = 0;
	other.sampler = 0;
}

Material::~Material()
{
	glDeleteTextures(1, &texture);
	glDeleteSamplers(1, &sampler);
}

void Material::BindMaterial()
{
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D, texture);
	glBindSampler(T_COLOR, sampler);
}