#include "WindowInput.h"
#include "Camera.h"
#include "Macros.h"
#include <GLFW/glfw3.h>

#include "IEUtility\IEMath.h"
#include "IEUtility\IEQuaternion.h"

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

//static const IEQuaternion mentalRotationX = IEQuaternion(IEMath::ToRadians(-18.0f), IEVector3::Xaxis);
//static const IEQuaternion mentalRotationY = IEQuaternion(IEMath::ToRadians(92.4f), IEVector3::Yaxis);

// Cornell
const Camera WindowInput::savedCamera =
{
	75.0f,
	0.15f,
	600.0f,
	1280,
	720,
	// Nyra Compare
	IEVector3(1.31693566f, 33.9894409f, 43.0923386f),
	IEVector3(2.25959873f, -2.40627909f, -185.163040f),
	// Sponza
	//IEVector3(-194f, 108.5f, 2.49f),
	//IEVector3(25.6f, 38.42f, -14.37f),
	// Cornell
	//IEVector3(-295.594666f, 135.253632f, 47.2294273f),
	//IEVector3(-72.7732697f, 87.5113068f, 8.60756302f),
	// Sponza 2
	//IEVector3(-182.487885f, 105.104980f, -18.5853291f),
	//IEVector3(30.7673531f, 16.5779095f, -8.26121616f),
	// Sponza 3
	//IEVector3(-112.188194f, 29.4227695f, -1.11264169f),
	//IEVector3(82.5518875f, 13.7036791f, 122.393143f),
	// Sibernik
	//IEVector3(-101.546936f, 52.9280930f, -4.32125282f),
	//IEVector3(125.090240f, 10.8526154f, -21.3083096f),
    // Mental Ray
    //IEVector3(171.137924, 120.530289, -23.6208248),
    //IEVector3(-20.9670410, 16.1353531, -16.5416222),
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
    , moveLight(false)
    , movement(false)
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

bool WindowInput::MoveLight() const
{
    return moveLight;
}

bool WindowInput::Movement() const
{
    return movement;
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
        case GLFW_KEY_KP_8:
            movement = !movement;
            if(movement) GI_LOG("Movement On");
            else GI_LOG("Movement Off");
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
        case GLFW_KEY_KP_5:
            moveLight = !moveLight;
            if(moveLight) GI_LOG("Move Light On");
            else GI_LOG("Move Light Off");
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