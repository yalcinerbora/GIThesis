#include "WindowInput.h"
#include "Camera.h"

void WindowInput::WindowPosChangedFunc(int x, int y)
{
	
}

void WindowInput::WindowFBChangedFunc(int width, int height)
{
	camera.width = static_cast<float>(width);
	camera.height = static_cast<float>(height);
}

void WindowInput::WindowSizeChangedFunc(int width, int height)
{
}

void WindowInput::WindowClosedFunc()
{

}

void WindowInput::WindowRefreshedFunc()
{

}

void WindowInput::WindowFocusedFunc(bool focused)
{

}

void WindowInput::WindowMinimizedFunc(bool minimized)
{

}

void WindowInput::KeyboardUsedFunc(int key, int, int action, int)
{
	//Numpad 4-6 to change between input schemes
	//Numpad 8-2 to change betweeen solutions


}

void WindowInput::MouseMovedFunc(double x, double y)
{
	
}

void WindowInput::MousePressedFunc(int button, int action, int)
{
	
}

void WindowInput::MouseScrolledFunc(double xOffset, double)
{
}