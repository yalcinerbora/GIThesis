#include "VoxFramebuffer.h"
#include "GLHeader.h"
#include <cassert>

VoxFramebuffer::VoxFramebuffer(GLsizei width, GLsizei height)
	: renderBuffers({0, 0})
	, fbo(0)
{
	glGenRenderbuffers(2, renderBuffers.data());
	glGenFramebuffers(1, &fbo);

	// Color
	glBindRenderbuffer(GL_RENDERBUFFER, renderBuffers[0]);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, 16, GL_RGBA8, width, height);

	glBindRenderbuffer(GL_RENDERBUFFER, renderBuffers[1]);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, 16, GL_DEPTH_COMPONENT24, width, height);

	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderBuffers[0]);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, renderBuffers[1]);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
}

VoxFramebuffer::~VoxFramebuffer()
{
	glDeleteRenderbuffers(2, renderBuffers.data());
	glDeleteFramebuffers(1, &fbo);	
}

void VoxFramebuffer::Bind()
{
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
}