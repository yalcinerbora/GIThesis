#include "Globals.h"

TwType twIEVector3Type;

const GLuint DeviceOGLParameters::uboAlignment;
const GLuint DeviceOGLParameters::ssboAlignment;

size_t DeviceOGLParameters::SSBOAlignOffset(size_t offset)
{
	if(offset % ssboAlignment == 0) return offset;
	return offset + (ssboAlignment - offset % ssboAlignment);
}

size_t DeviceOGLParameters::UBOAlignOffset(size_t offset)
{
	if(offset % uboAlignment == 0) return offset;
	return offset + (uboAlignment - offset % uboAlignment);
}