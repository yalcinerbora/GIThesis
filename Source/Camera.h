

/**

Globals For Rendering

ATM only Camera

*/


#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEVector3.h"
#include "FrameTransformBuffer.h"

struct Camera
{
	// Perspective Projection Params
	float						fov;
	float						near;
	float						far;

	// Viewport Params
	float						width;
	float						height;

	// Camera Orientation
	IEVector3					pos;
	IEVector3					at;
	IEVector3					up;

	FrameTransformBufferData	generateTransform() const
	{
		return
		{
			IEMatrix4x4::LookAt(pos, at, up),
			IEMatrix4x4::Perspective(fov, width / height, near, far),
			IEMatrix4x4::IdentityMatrix
		};
	}
};
#endif //__CAMERA_H__