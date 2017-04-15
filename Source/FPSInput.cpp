#include "FPSInput.h"
#include "IEUtility/IEVector3.h"
#include "IEUtility/IEQuaternion.h"
#include "Camera.h"
#include "GLFW/glfw3.h"

const std::string FPSInput::FPSInputName = "FPSInput";

FPSInput::FPSInput(double sensitivity,
				   double moveRatio,
				   double moveRatioModifier)
	: sensitivity(sensitivity)
	, moveRatio(moveRatio)
	, moveRatioModifier(moveRatioModifier)
	, fpsMode(false)
	, mouseX(0.0)
	, mouseY(0.0)
	, moveRatioModified(moveRatio)
{}

void FPSInput::KeyboardUsedFunc(Camera& camera, int key, int osKey, int action, int modifier)
{

	// Shift modifier
	if(action == GLFW_PRESS && key == GLFW_KEY_LEFT_SHIFT)
	{
		moveRatioModified = moveRatio * 0.25;
	}
	else if(action == GLFW_RELEASE  && key == GLFW_KEY_LEFT_SHIFT)
	{
		moveRatioModified = moveRatio;
	}

	// Movement
	if(!(action == GLFW_RELEASE))
	{
		IEVector3 lookDir = (camera.centerOfInterest - camera.pos).NormalizeSelf();
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		switch(key)
		{

			case GLFW_KEY_W:
				camera.pos += lookDir * static_cast<float>(moveRatioModified);
				camera.centerOfInterest += lookDir * static_cast<float>(moveRatioModified);
				break;
			case GLFW_KEY_A:

				camera.pos += side * static_cast<float>(moveRatioModified);
				camera.centerOfInterest += side * static_cast<float>(moveRatioModified);
				break;
			case GLFW_KEY_S:
				camera.pos += lookDir * static_cast<float>(-moveRatioModified);
				camera.centerOfInterest += lookDir * static_cast<float>(-moveRatioModified);
				break;
			case GLFW_KEY_D:
				camera.pos += side * static_cast<float>(-moveRatioModified);
				camera.centerOfInterest += side * static_cast<float>(-moveRatioModified);
				break;

			default:
				break;
		}
	}
}

void FPSInput::MouseMovedFunc(Camera& camera, double x, double y)
{
	// Check with latest recorded input
	double diffX = x - mouseX;
	double diffY = y - mouseY;

	if(fpsMode)
	{
		// X Rotation
		IEVector3 lookDir = camera.centerOfInterest - camera.pos;
		IEQuaternion rotateX(static_cast<float>(-diffX * sensitivity), IEVector3::Yaxis);
		IEVector3 rotated = rotateX.ApplyRotation(lookDir);
		camera.centerOfInterest = camera.pos + rotated;

		// Y Rotation
		lookDir = camera.centerOfInterest - camera.pos;
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		IEQuaternion rotateY(static_cast<float>(diffY * sensitivity), side);
		rotated = rotateY.ApplyRotation((lookDir));
		camera.centerOfInterest = camera.pos + rotated;

		// Redefine up
		// Enforce an up vector which is ortogonal to the xz plane
		camera.up = rotated.CrossProduct(side);
		camera.up.setZ(0.0f);
		camera.up.setX(0.0f);
		camera.up.NormalizeSelf();
	}
	mouseX = x;
	mouseY = y;
}

void FPSInput::MousePressedFunc(Camera& camera, int button, int action, int modifier)
{
	switch(button)
	{
		case GLFW_MOUSE_BUTTON_LEFT:
			fpsMode = (action == GLFW_RELEASE) ? false : true;
			break;
	}
}

void FPSInput::MouseScrolledFunc(Camera& camera, double xOffset, double yOffset)
{}

const std::string& FPSInput::Name() const
{
	return FPSInputName;
}