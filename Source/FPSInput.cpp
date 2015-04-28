#include "FPSInput.h"
#include "IEUtility/IEVector3.h"
#include "IEUtility/IEQuaternion.h"
#include "Camera.h"
#include "GLFW/glfw3.h"

double FPSInput::Sensitivity = 0.005f;
double FPSInput::MoveRatio = 8.00f;

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
{}

void FPSInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{
	WindowInput::KeyboardUsedFunc(key, osKey, action, modifier);

	IEVector3 lookDir = (camera.pos - camera.at).NormalizeSelf();
	IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
	switch(key)
	{
		
		case GLFW_KEY_W:
			camera.pos += lookDir * static_cast<float>(-MoveRatio);
			camera.at += lookDir * static_cast<float>(-MoveRatio);
			break;
		case GLFW_KEY_A:
			
			camera.pos += side * static_cast<float>(-MoveRatio);
			camera.at += side * static_cast<float>(-MoveRatio);
			break;
		case GLFW_KEY_S:
			camera.pos += lookDir * static_cast<float>(MoveRatio);
			camera.at += lookDir * static_cast<float>(MoveRatio);
			break;
		case GLFW_KEY_D:
			camera.pos += side * static_cast<float>(MoveRatio);
			camera.at += side * static_cast<float>(MoveRatio);
			break;

		default:
			break;
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
		IEVector3 lookDir = camera.pos - camera.at;
		IEQuaternion rotateX(static_cast<float>(-diffX * Sensitivity), IEVector3::Yaxis);
		IEVector3 rotated = rotateX.ApplyRotation(lookDir);
		camera.at = camera.pos - rotated;

		// Y Rotation
		lookDir = camera.pos - camera.at;
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		IEQuaternion rotateY(static_cast<float>(diffY * Sensitivity), side);
		rotated = rotateY.ApplyRotation((lookDir));
		camera.at = camera.pos - rotated;

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