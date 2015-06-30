#include "WindowInput.h"
#include "Camera.h"
#include "Macros.h"
#include <GLFW/glfw3.h>

WindowInput::WindowInput(Camera& cam,
						 uint32_t& currentSolution,
						 uint32_t& currentScene,
						 uint32_t& currentInput)
	: camera(cam)
	, currentSolution(currentSolution)
	, currentScene(currentScene)
	, currentInput(currentInput)
{
}

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

void WindowInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{	
	if(action == GLFW_RELEASE)
	switch(key)
	{
		// Solution Change
		case GLFW_KEY_KP_7:
			GI_LOG("Changing Solution");
			currentSolution++;
			break;
		case GLFW_KEY_KP_9:
			GI_LOG("Changing Solution");
			currentSolution--;
			break;

		// Scene Change
		case GLFW_KEY_KP_4:
			currentScene++;
			GI_LOG("Changing Scene");
			break;
		case GLFW_KEY_KP_6:
			GI_LOG("Changing Scene");
			currentScene--;
			break;

		// Input Schemes
		case GLFW_KEY_KP_1:
			GI_LOG("Changing Input Scheme");
			currentInput++;
			break;
		case GLFW_KEY_KP_3:
			GI_LOG("Changing Input Scheme");
			currentInput--;
			break;

		default:
			break;
	}
}

void WindowInput::MouseMovedFunc(double x, double y)
{
}

void WindowInput::MousePressedFunc(int button, int action, int modifier)
{
}

void WindowInput::MouseScrolledFunc(double xOffset, double yOffset)
{
}