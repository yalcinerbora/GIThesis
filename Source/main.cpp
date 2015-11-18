#include <iostream>
#include <GFG/GFGHeader.h>
#include <AntTweakBar.h>

#include "Window.h"

#include "NoInput.h"
#include "FPSInput.h"
#include "MayaInput.h"

#include "DeferredRenderer.h"

#include "EmptyGISolution.h"
#include "ThesisSolution.h"

#include "Globals.h"
#include "Camera.h"
#include "Scene.h"
#include "Macros.h"
#include "CudaInit.h"

#include "IEUtility/IEMath.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IETimer.h"
#include <GLFW/glfw3.h>

int main()
{
	// Cuda Init
	CudaInit::InitCuda();

	Camera mainRenderCamera =
	{
		90.0f,
		0.1f,
		600.0f,
		1280,
		720,
		{ -180.0f, 145.0f, 0.3f },
		IEVector3::ZeroVector,
		IEVector3::Yaxis
	};

	uint32_t currentSolution = 0, currentScene = 0, currentInputScheme = 0, oldSolution = currentSolution;
	std::vector<SolutionI*>	solutions;
	std::vector<SceneI*>	scenes;
	std::vector<InputManI*>	inputSchemes;

	// Input Schemes
	NoInput nullInput(mainRenderCamera, currentSolution, currentScene, currentInputScheme);
	MayaInput mayaInput(mainRenderCamera, currentSolution, currentScene, currentInputScheme);
	FPSInput fpsInput(mainRenderCamera, currentSolution, currentScene, currentInputScheme);
	inputSchemes.push_back(&nullInput);
	inputSchemes.push_back(&mayaInput);
	inputSchemes.push_back(&fpsInput);

	// Window Init
	WindowProperties winProps
	{
		1280,
		720,
		WindowScreenType::WINDOWED
	};
	Window mainWindow(nullInput,
					  winProps);

	// DeferredRenderer
	DeferredRenderer deferredRenderer;

	// Scenes
	Light sponzaLights[] = 
	{
		// Directional Light
		// White Color
		// 1-2 PM Sunlight direction (if you consider lionhead(window) is at north)
		{
			{ 0.0f, 0.0f, 0.0f, static_cast<float>(LightType::DIRECTIONAL) },
			{ 0.0f, -IEMath::CosF(IEMath::ToRadians(9.5f)), -IEMath::SinF(IEMath::ToRadians(9.5f)), std::numeric_limits<float>::infinity()},
			IEVector4(1.0f, 1.0f, 1.0f, 4.2f)
		},
		//Point Lights
		//Various Colors color effecting radius 60 units
		{
			{ 212.6f, 50.8f, -85.3f, static_cast<float>(LightType::POINT) },
			{ 0.0f, 0.0f, 0.0f, 120.0f },
			IEVector4(1.0f, 1.0f, 1.0f, 3000.0f)
		},
		{
			{ -116.8f, 27.5f, 17.0f, static_cast<float>(LightType::POINT) },
			{ 0.0f, 0.0f, 0.0f, 120.0f },
			IEVector4(1.0f, 1.0f, 1.0f, 3000.0f)
		},
		{
			{ 92.2f, 25.9f, 16.2f, static_cast<float>(LightType::POINT) },
			{ 0.0f, 0.0f, 0.0f, 120.0f },
			IEVector4(1.0f, 1.0f, 1.0f, 3000.0f)
		}
	};
	Light cornellLights[] =
	{
		// Area Light
		// White Color
		// Top of the room (center of white rectangle)
		// Square light
		// Covers Entire Room
		{
			{ 0.0f, 160.0f, 0.0f, static_cast<float>(LightType::POINT)},
			{ 0.0f, -1.0f, 0.0f, 3300.0f},
			IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		}
	};

	Scene crySponza(Scene::sponzaFileName, {sponzaLights, 4}, 2.7f, 5.1f);
	Scene cornellBox(Scene::cornellboxFileName, {cornellLights, 1}, 0.513f, 4.1f);
	Scene movingObjects(Scene::movingObjectsFileName, {sponzaLights, 4}, 0.513f, 0.03f);
	scenes.push_back(&crySponza);
	scenes.push_back(&cornellBox);
	scenes.push_back(&movingObjects);

	// Solutions
	EmptyGISolution emptySolution(deferredRenderer);
	ThesisSolution thesisSolution(deferredRenderer, mainRenderCamera.pos);
	solutions.push_back(&emptySolution);
	solutions.push_back(&thesisSolution);

	// Main Help
	TwDefine(" GLOBAL iconpos=tl ");
	TwDefine(" GLOBAL help='GI Implementation using voxels.\n"
			 "\tUse NumPad 7,8 to change between solutions.\n"
			 "\tUse NumPad 4,6 to change between scenes.\n"
			 "\tUse NumPad 1,3 to change between camera input schemes.\n"
			 "\t\t Input Scheme#1 : No Input.\n"
			 "\t\t Input Scheme#2 : Maya Input. (MouseBTN1 to rotate around COI. Mouse BTN3 to translate COI)\n"
			 "\t\t Input Scheme#3 : FPS Input. (WASD to move MouseBTN1 to look around)\n"
			 "' ");

	// FPS Timer
	IETimer t;
	t.Start();

	// All Init
	// Render Loop
	//float angle = 0.0f;
	//float posXInc = 0.0f;
	while(!mainWindow.WindowClosed())
	{
		// Constantly Check Input Scheme Change
		mainWindow.ChangeInputScheme(*inputSchemes[currentInputScheme % inputSchemes.size()]);

		// Enforce intialization if solution changed
		bool forceInit = false;
		if(oldSolution != currentSolution)
		{
			forceInit = true;
			solutions[oldSolution % solutions.size()]->Release();
			oldSolution = currentSolution;
		}
			
		SolutionI* solution = solutions[currentSolution % solutions.size()];
		if(!solution->IsCurrentScene(*scenes[currentScene % scenes.size()]) ||
		   forceInit)
		{
			solutions[oldSolution % solutions.size()]->Release();
			solution->Init(*scenes[currentScene % scenes.size()]);
		}

		//// Rotation
		//IEVector3 dir = sponzaLights[0].direction;
		//angle += 0.00005f;
		//IEQuaternion rot(angle, IEVector3::Xaxis);
		//dir = rot.ApplyRotation(dir);
		//scenes[currentScene]->getSceneLights().ChangeLightDir(0, dir.Normalize());
		//
		//IEVector3 pos = sponzaLights[2].position;
		//posXInc += 0.05f;
		//pos.setX(pos.getX() + posXInc);
		//scenes[currentScene]->getSceneLights().ChangeLightPos(2, pos);

		// Render frame
		solution->Frame(mainRenderCamera);
		
		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();

		t.Lap();
		solution->SetFPS(t.ElapsedMilliS());
	}
	return 0;
}