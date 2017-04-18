#include "Globals.h"

TwType twIEVector3Type;

GLint DeviceOGLParameters::uboAlignment = 4;
GLint DeviceOGLParameters::ssboAlignment = 4;

size_t DeviceOGLParameters::SSBOAlignOffset(size_t offset)
{
	return AlignOffset(offset, ssboAlignment);
}

size_t DeviceOGLParameters::UBOAlignOffset(size_t offset)
{
	return AlignOffset(offset, uboAlignment);
}

size_t DeviceOGLParameters::AlignOffset(size_t offset, size_t alignment)
{
	if(offset % alignment == 0) return offset;
	return offset + (alignment - offset % alignment);
}