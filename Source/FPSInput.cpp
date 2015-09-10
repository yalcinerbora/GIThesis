#include "FPSInput.h"
#include "IEUtility/IEVector3.h"
#include "IEUtility/IEQuaternion.h"
#include "Camera.h"
#include "GLFW/glfw3.h"

double FPSInput::Sensitivity = 0.005;
double FPSInput::MoveRatio = 4.30;

FPSInput::FPSInput(Camera& cam,
					 uint32_t& currentSolution,
					 uint32_t& currentScene,
					 uint32_t& currentInput)
	: WindowInput(cam,
				  currentSolution,
				  currentScene,
				  currentInput)
	, FPSMode(false)
	, mouseX(0.0)
	, mouseY(0.0)
	, moveRatioModified(MoveRatio)
{}

void FPSInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{
	WindowInput::KeyboardUsedFunc(key, osKey, action, modifier);

	// Shift modifier
	if(action == GLFW_PRESS && key == GLFW_KEY_LEFT_SHIFT)
	{
		moveRatioModified = MoveRatio * 0.25;
	}
	else if(action == GLFW_RELEASE  && key == GLFW_KEY_LEFT_SHIFT)
	{
		moveRatioModified = MoveRatio;
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

void FPSInput::MouseMovedFunc(double x, double y)
{
	WindowInput::MouseMovedFunc(x, y);

	// Check with latest recorded input
	double diffX = x - mouseX;
	double diffY = y - mouseY;

	WindowInput::MouseMovedFunc(x, y);
	if(FPSMode)
	{
		// X Rotation
		IEVector3 lookDir = camera.centerOfInterest - camera.pos;
		IEQuaternion rotateX(static_cast<float>(-diffX * Sensitivity), IEVector3::Yaxis);
		IEVector3 rotated = rotateX.ApplyRotation(lookDir);
		camera.centerOfInterest = camera.pos + rotated;

		// Y Rotation
		lookDir = camera.centerOfInterest - camera.pos;
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		IEQuaternion rotateY(static_cast<float>(diffY * Sensitivity), side);
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

void FPSInput::MousePressedFunc(int button, int action, int modifier)
{
	WindowInput::MousePressedFunc(button, action, modifier);

	switch(button)
	{
		case GLFW_MOUSE_BUTTON_LEFT:
			FPSMode = (action == GLFW_RELEASE) ? false : true;
			break;
	}
}

void FPSInput::MouseScrolledFunc(double xOffset, double yOffset)
{
	WindowInput::MouseScrolledFunc(xOffset, yOffset);
}