
#include "WindowInput.h"
#include "Camera.h"
#include "Macros.h"

#include "SolutionI.h"
#include "SceneI.h"

#include "IEUtility\IEMath.h"
#include "IEUtility\IEQuaternion.h"
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

//static const IEQuaternion mentalRotationX = IEQuaternion(IEMath::ToRadians(-18.0f), IEVector3::Xaxis);
//static const IEQuaternion mentalRotationY = IEQuaternion(IEMath::ToRadians(92.4f), IEVector3::Yaxis);

// Cornell
//const Camera WindowInput::savedCamera =
//{
//	75.0f,
//	0.15f,
//	600.0f,
//	1280,
//	720,
//	// Nyra Compare
//	IEVector3(1.31693566f, 33.9894409f, 43.0923386f),
//	IEVector3(2.25959873f, -2.40627909f, -185.163040f),
//	// Sponza
//	//IEVector3(-194f, 108.5f, 2.49f),
//	//IEVector3(25.6f, 38.42f, -14.37f),
//	// Cornell
//	//IEVector3(-295.594666f, 135.253632f, 47.2294273f),
//	//IEVector3(-72.7732697f, 87.5113068f, 8.60756302f),
//	// Sponza 2
//	//IEVector3(-182.487885f, 105.104980f, -18.5853291f),
//	//IEVector3(30.7673531f, 16.5779095f, -8.26121616f),
//	// Sponza 3
//	//IEVector3(-112.188194f, 29.4227695f, -1.11264169f),
//	//IEVector3(82.5518875f, 13.7036791f, 122.393143f),
//	// Sibernik
//	//IEVector3(-101.546936f, 52.9280930f, -4.32125282f),
//	//IEVector3(125.090240f, 10.8526154f, -21.3083096f),
//    // Mental Ray
//    //IEVector3(171.137924, 120.530289, -23.6208248),
//    //IEVector3(-20.9670410, 16.1353531, -16.5416222),
//	IEVector3::Yaxis
//};

WindowInput::WindowInput(Camera& camera,
						 const std::vector<CameraInputI*>& cameraInputs,
						 const std::vector<SolutionI*>& solutions,
						 const std::vector<SceneI*>& scenes)
	: camera(camera)
	, cameraInputs(cameraInputs)
	, scenes(scenes)
	, solutions(solutions)
	, currentCameraInput(0)
	, currentSolution(0)
	, currentScene(0)
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

void WindowInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{	
	if(action == GLFW_RELEASE)
	switch(key)
	{
		// Solution Change
		case GLFW_KEY_KP_7:
			GI_LOG("Changing Solution");
			solutions[currentSolution]->Release();
			currentSolution = (currentSolution == 0)
								? (solutions.size() - 1)
								: (currentSolution - 1);
			solutions[currentSolution]->Load(*scenes[currentScene]);
			GI_LOG("Solution \"%s\"", solutions[currentSolution]->Name().c_str());
			break;
		case GLFW_KEY_KP_9:
			solutions[currentSolution]->Release();
			currentSolution = (currentSolution == solutions.size() - 1)
								? (0)
								: (currentSolution + 1);
			solutions[currentSolution]->Load(*scenes[currentScene]);
			GI_LOG("Solution \"%s\"", solutions[currentSolution]->Name().c_str());
			break;

		// Scene Change
		case GLFW_KEY_KP_4:
			solutions[currentSolution]->Release();
			scenes[currentScene]->Release();
			currentScene = (currentScene == 0)
								? (scenes.size() - 1)
								: (currentScene - 1);
			scenes[currentScene]->Load();
			solutions[currentSolution]->Load(*scenes[currentScene]);
			GI_LOG("Scene \"%s\"", scenes[currentScene]->Name().c_str());
			break;
		case GLFW_KEY_KP_6:
			solutions[currentSolution]->Release();
			scenes[currentScene]->Release();
			currentScene = (currentScene == scenes.size() - 1)
								? (0)
								: (currentScene + 1);
			scenes[currentScene]->Load();
			solutions[currentSolution]->Load(*scenes[currentScene]);
			GI_LOG("Scene \"%s\"", scenes[currentScene]->Name().c_str());
			break;

		// Input Schemes
		case GLFW_KEY_KP_1:
			currentCameraInput = (currentCameraInput == 0)
								? (cameraInputs.size() - 1)
								: (currentCameraInput - 1);
			GI_LOG("Input Scheme \"%s\"", cameraInputs[currentCameraInput]->Name().c_str());
			break;
		case GLFW_KEY_KP_3:			
			currentCameraInput = (currentCameraInput == cameraInputs.size() - 1)
								? (0)
								: (currentCameraInput + 1);
			GI_LOG("Input Scheme \"%s\"", cameraInputs[currentCameraInput]->Name().c_str());
			break;
	}

	// Handle Callbacks
	// Call functions that has same key action combination
	auto range = callbacks.equal_range(std::make_pair(key, action));
	for(auto it = range.first; it != range.second; ++it)
	{
		// Call Those Functions
		it->second();	
	}
}

void WindowInput::MouseMovedFunc(double x, double y)
{
	cameraInputs[currentCameraInput]->MouseMovedFunc(camera, x, y);
}

void WindowInput::MousePressedFunc(int button, int action, int modifier)
{
	cameraInputs[currentCameraInput]->MousePressedFunc(camera, button, action, modifier);
}

void WindowInput::MouseScrolledFunc(double xOffset, double yOffset)
{
	cameraInputs[currentCameraInput]->MouseScrolledFunc(camera, xOffset, yOffset);
}

SolutionI* WindowInput::Solution()
{
	return solutions[currentSolution];
}
SceneI* WindowInput::Scene()
{
	return scenes[currentScene];
}
