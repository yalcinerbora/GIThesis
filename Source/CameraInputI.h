#pragma once
/**

CameraInputI

Proivdes Classes for Camera Movement
*/

#include <string>

struct Camera;

// TODO: Add Joystick later
class CameraInputI
{
	public:
		virtual						~CameraInputI() = default;
		virtual void				KeyboardUsedFunc(Camera&, int, int, int, int) = 0;
		virtual void				MouseMovedFunc(Camera&, double, double) = 0;
		virtual void				MousePressedFunc(Camera&, int, int, int) = 0;
		virtual void				MouseScrolledFunc(Camera&, double, double) = 0;
		
		virtual const std::string&	Name() const = 0;
};