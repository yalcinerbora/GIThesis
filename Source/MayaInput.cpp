#include "MayaInput.h"
#include "Camera.h"
#include "IEUtility/IEQuaternion.h"
#include "GLFW/glfw3.h"
#include "Macros.h"

double MayaInput::Sensitivity = 0.005;
double MayaInput::ZoomPrecentage = 0.1;	// reduce by this amount
double MayaInput::TranslateModifier = 0.2;

MayaInput::MayaInput(Camera& cam,
					uint32_t& currentSolution,
					uint32_t& currentScene,
					uint32_t& currentInput)
	: WindowInput(cam, 
				  currentSolution, 
				  currentScene, 
				  currentInput)
	, moveMode(false)
	, translateMode(false)
	, mouseX(0.0)
	, mouseY(0.0)
{}

void MayaInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{
	WindowInput::KeyboardUsedFunc(key, osKey, action, modifier);
}

void MayaInput::MouseMovedFunc(double x, double y)
{
	// Check with latest recorded input
	double diffX = x - mouseX;
	double diffY = y - mouseY;

	WindowInput::MouseMovedFunc(x, y);
	if(moveMode)
	{
		// X Rotation
		IEVector3 lookDir = camera.pos - camera.at;
		IEQuaternion rotateX(static_cast<float>(diffX * Sensitivity), IEVector3::Yaxis);
		IEVector3 rotated = rotateX.ApplyRotation(lookDir);
		camera.pos = rotated + camera.at;

		// Y Rotation
		lookDir = camera.pos - camera.at;
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		IEQuaternion rotateY(static_cast<float>(diffY * Sensitivity), side);
		rotated = rotateY.ApplyRotation((lookDir));
		camera.pos = rotated + camera.at;

		// Redefine up
		// Enforce an up vector which is ortogonal to the xz plane
		camera.up = rotated.CrossProduct(side);
		camera.up.setZ(0.0f);
		camera.up.setX(0.0f);
		camera.up.NormalizeSelf();
	}
	if(translateMode)
	{
		IEVector3 lookDir = camera.pos - camera.at;
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		camera.pos += static_cast<float>(diffX * TranslateModifier) * -side;
		camera.at += static_cast<float>(diffX * TranslateModifier) * -side;

		camera.pos += static_cast<float>(diffY * TranslateModifier) * -camera.up;
		camera.at += static_cast<float>(diffY * TranslateModifier) * -camera.up;
	}

	mouseX = x;
	mouseY = y;
}

void MayaInput::MousePressedFunc(int button, int action, int modifier)
{
	WindowInput::MousePressedFunc(button, action, modifier);

	switch(button)
	{
		case GLFW_MOUSE_BUTTON_LEFT:
			moveMode = (action == GLFW_RELEASE) ? false : true;
			break;
		case GLFW_MOUSE_BUTTON_MIDDLE:
			translateMode = (action == GLFW_RELEASE) ? false : true;
			break;
	}
}

void MayaInput::MouseScrolledFunc(double xOffset, double yOffset)
{
	WindowInput::MouseScrolledFunc(xOffset, yOffset);

	// Zoom to the focus until some threshold
	IEVector3 lookDir = camera.pos - camera.at;
	lookDir *= static_cast<float>(1.0 - yOffset * ZoomPrecentage);
	if(lookDir.Length() > 0.1f)
		camera.pos = lookDir + camera.at;
}