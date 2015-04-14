/**

Globals For Rendering

ATM only Camera

*/


#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEVector3.h"

struct Camera
{
	IEMatrix4x4		perspectiveProjection;

	// Camera Orientation
	IEVector3		pos;
	IEVector3		at;
	IEVector3		up;
};

static Camera mainRenderCamera = 
{
	IEMatrix4x4::IdentityMatrix,
	IEVector3::ZeroVector,
	IEVector3(0.0f, 0.0f, 1.0f),
	IEVector3::Yaxis
};
#endif //__NOINPUT_H__