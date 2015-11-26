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

void WindowInput::AddKeyCallback(int key, int action, void(*func)(void*), void* ptr)
{
	callbacks.emplace(std::make_pair(key, action), std::make_pair(func, ptr));
}

void WindowInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{	
	if(action == GLFW_RELEASE)
	switch(key)
	{
		// Solution Change
		case GLFW_KEY_KP_7:
			GI_LOG("Changing Solution");
			currentSolution--;
			break;
		case GLFW_KEY_KP_9:
			GI_LOG("Changing Solution");
			currentSolution++;
			break;

		// Scene Change
		case GLFW_KEY_KP_4:
			currentScene--;
			GI_LOG("Changing Scene");
			break;
		case GLFW_KEY_KP_6:
			GI_LOG("Changing Scene");
			currentScene++;
			break;

		// Input Schemes
		case GLFW_KEY_KP_1:
			GI_LOG("Changing Input Scheme");
			currentInput--;
			break;
		case GLFW_KEY_KP_3:
			GI_LOG("Changing Input Scheme");
			currentInput++;
			break;

		default:
			break;
	}

	// Handle Callbacks
	// Call functions that has same key action combination
	auto range = callbacks.equal_range(std::make_pair(key, action));
	for(auto it = range.first; it != range.second; ++it)
	{
		it->second.first(it->second.second);	
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