

/**

Globals For Rendering

ATM only Camera

*/


#ifndef __CAMERA_H__
#define __CAMERA_H__

#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEVector3.h"
#include "Globals.h"

struct Camera
{
	// Perspective Projection Params
	float						fovX;
	float						near;
	float						far;

	// Viewport Params
	float						width;
	float						height;

	// Camera Orientation
	IEVector3					pos;
	IEVector3					centerOfInterest;
	IEVector3					up;

	FrameTransformData			GenerateTransform() const
	{
		return
		{
			IEMatrix4x4::LookAt(pos, centerOfInterest, up),
			IEMatrix4x4::Perspective(fovX, width / height, near, far)
		};
	}
};
#endif //__CAMERA_H__