#include "FrameTransformBuffer.h"
#include "Globals.h"

FrameTransformBuffer::FrameTransformBuffer()
	: bufferId(0)
{
	glGenBuffers(1, &bufferId);
	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glBufferData(GL_COPY_WRITE_BUFFER, sizeof(FrameTransformBufferData),
					nullptr,
					GL_DYNAMIC_DRAW);
}

FrameTransformBuffer::~FrameTransformBuffer()
{
	glDeleteBuffers(1, &bufferId);
}

void FrameTransformBuffer::Update(FrameTransformBufferData ftd)
{
	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glBufferSubData(GL_COPY_WRITE_BUFFER, 0, sizeof(FrameTransformBufferData),
					&ftd);
}
void FrameTransformBuffer::Bind()
{
	glBindBufferBase(GL_UNIFORM_BUFFER, U_FTRANSFORM, bufferId);
}