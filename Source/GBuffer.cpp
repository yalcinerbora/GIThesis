#include "GBuffer.h"
#include "GLHeader.h"
#include <cassert>

GBuffer::GBuffer(GLuint w, GLuint h)
 : fbo(0)
 , fboTexSampler(0)
 , width(w)
 , height(h)
 , depthR32FCopy(0)
{
	// Generate Textures
	glGenTextures(3, rtTextures);

	// Color Tex
	glBindTexture(GL_TEXTURE_2D, rtTextures[static_cast<int>(RenderTargetLocation::COLOR)]);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);

	// Normal Tex
	glBindTexture(GL_TEXTURE_2D, rtTextures[static_cast<int>(RenderTargetLocation::NORMAL)]);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG16UI, width, height);
	
	glBindTexture(GL_TEXTURE_2D, rtTextures[static_cast<int>(RenderTargetLocation::DEPTH)]);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH_COMPONENT32F, width, height);

	//depthR32FView (Cuda compatiblity)
	glGenTextures(1, &depthR32FCopy);
	glBindTexture(GL_TEXTURE_2D, depthR32FCopy);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, width, height);

	// Sampler
	glGenSamplers(1, &fboTexSampler);
	glSamplerParameteri(fboTexSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(fboTexSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

	// Generate FBO
	glGenFramebuffers(1, &fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
						   GL_TEXTURE_2D, rtTextures[static_cast<int>(RenderTargetLocation::COLOR)],
						   0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
						   GL_TEXTURE_2D, rtTextures[static_cast<int>(RenderTargetLocation::NORMAL)],
						   0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
						   GL_TEXTURE_2D, rtTextures[static_cast<int>(RenderTargetLocation::DEPTH)],
						   0);

	const GLenum db[] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
	glDrawBuffers(2, db);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

GBuffer::~GBuffer()
{
	glDeleteFramebuffers(1, &fbo);
	glDeleteSamplers(1, &fboTexSampler);
	glDeleteTextures(3, rtTextures);
	glDeleteTextures(1, &depthR32FCopy);
}

void GBuffer::BindAsTexture(GLuint texTarget,
							RenderTargetLocation loc)
{
	glActiveTexture(GL_TEXTURE0 + texTarget);
	glBindSampler(texTarget, fboTexSampler);
	glBindTexture(GL_TEXTURE_2D, rtTextures[static_cast<GLuint>(loc)]);
}

void GBuffer::BindAsFBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}

void GBuffer::AlignViewport()
{
	glViewport(0, 0, width, height);
}

GLuint GBuffer::getDepthGL()
{
	return rtTextures[static_cast<int>(RenderTargetLocation::DEPTH)];
}

GLuint GBuffer::getNormalGL()
{
	return rtTextures[static_cast<int>(RenderTargetLocation::NORMAL)];
}

GLuint GBuffer::getDepthGLView()
{
	return depthR32FCopy;
}

void GBuffer::BindDefaultFBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

//void GBuffer::CopyDepthToCudaMapped()
//{
//}