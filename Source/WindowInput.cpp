#include "WindowInput.h"
#include "Camera.h"
#include "Macros.h"
#include <GLFW/glfw3.h>

//// Sponza
//const Camera WindowInput::savedCamera = 
//{
//	90.0f,
//	0.15f,
//	600.0f,
//	1280,
//	720,
//	IEVector3(-194, 108.5, 2.49),
//	IEVector3(25.6, 38.42, -14.37),
//	IEVector3::Yaxis
//};

// Cornell
const Camera WindowInput::savedCamera =
{
	90.0f,
	0.15f,
	600.0f,
	1280,
	720,
	// Sponza
	//IEVector3(-194, 108.5, 2.49),
	//IEVector3(25.6, 38.42, -14.37),
	// Cornell
	//IEVector3(-295.594666, 135.253632, 47.2294273),
	//IEVector3(-72.7732697, 87.5113068, 8.60756302),
	// Sponza 2
	//IEVector3(-182.487885, 105.104980, -18.5853291),
	//IEVector3(30.7673531, 16.5779095, -8.26121616),
	// Sponza 3
	//IEVector3(-112.188194, 29.4227695, -1.11264169),
	//IEVector3(82.5518875, 13.7036791, 122.393143),
	// Sibernik
	IEVector3(-190.550354, 119.162132, 44.7449226),
	IEVector3(23.1233673, 46.8825302, -5.65195084),
	IEVector3::Yaxis
};

WindowInput::WindowInput(Camera& cam,
						 uint32_t& currentSolution,
						 uint32_t& currentScene,
						 uint32_t& currentInput)
	: camera(cam)
	, currentSolution(currentSolution)
	, currentScene(currentScene)
	, currentInput(currentInput)
{}

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
		case GLFW_KEY_KP_2:
			GI_LOG("Saved Camera");
			camera = savedCamera;
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