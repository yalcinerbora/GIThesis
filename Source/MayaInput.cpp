#include "MayaInput.h"
#include "Camera.h"
#include "IEUtility/IEQuaternion.h"
#include "GLFW/glfw3.h"
#include "Macros.h"

const std::string MayaInput::MayaInputName = "MayaInput";

MayaInput::MayaInput(double sensitivity,
					 double zoomPercentage,
					 double translateModifier)
	: sensitivity(sensitivity)
	, zoomPercentage(zoomPercentage)
	, translateModifier(translateModifier)
	, moveMode(false)
	, translateMode(false)
	, mouseX(0.0)
	, mouseY(0.0)
{}

void MayaInput::KeyboardUsedFunc(Camera& camera, int key, int osKey, int action, int modifier)
{}

void MayaInput::MouseMovedFunc(Camera& camera, double x, double y)
{
	// Check with latest recorded input
	double diffX = x - mouseX;
	double diffY = y - mouseY;

	if(moveMode)
	{
		// X Rotation
		IEVector3 lookDir = camera.centerOfInterest - camera.pos;
		IEQuaternion rotateX(static_cast<float>(-diffX * sensitivity), IEVector3::YAxis);
		IEVector3 rotated = rotateX.ApplyRotation(lookDir);
		camera.pos = camera.centerOfInterest - rotated;

		// Y Rotation
		lookDir = camera.centerOfInterest - camera.pos;
		IEVector3 left = camera.up.CrossProduct(lookDir).NormalizeSelf();
		IEQuaternion rotateY(static_cast<float>(diffY * sensitivity), left);
		rotated = rotateY.ApplyRotation((lookDir));
		camera.pos = camera.centerOfInterest - rotated;

		// Redefine up
		// Enforce an up vector which is ortogonal to the xz plane
		camera.up = rotated.CrossProduct(left);
		camera.up.setZ(0.0f);
		camera.up.setX(0.0f);
		camera.up.NormalizeSelf();
	}
	if(translateMode)
	{
		IEVector3 lookDir = camera.centerOfInterest - camera.pos;
		IEVector3 side = camera.up.CrossProduct(lookDir).NormalizeSelf();
		camera.pos += static_cast<float>(diffX * translateModifier) * side;
		camera.centerOfInterest += static_cast<float>(diffX * translateModifier) * side;

		camera.pos += static_cast<float>(diffY * translateModifier) * camera.up;
		camera.centerOfInterest += static_cast<float>(diffY * translateModifier) * camera.up;
	}

	mouseX = x;
	mouseY = y;
}

void MayaInput::MousePressedFunc(Camera& camera, int button, int action, int modifier)
{
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

void MayaInput::MouseScrolledFunc(Camera& camera, double xOffset, double yOffset)
{
	// Zoom to the focus until some threshold
	IEVector3 lookDir = camera.pos - camera.centerOfInterest;
	lookDir *= static_cast<float>(1.0 - yOffset * zoomPercentage);
	if(lookDir.Length() > 0.1f)
		camera.pos = lookDir + camera.centerOfInterest;
}

const std::string& MayaInput::Name() const
{
	return MayaInputName;
}