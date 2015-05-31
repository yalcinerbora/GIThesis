#include <iostream>
#include <GFG/GFGHeader.h>

#include "Window.h"

#include "NoInput.h"
#include "FPSInput.h"
#include "MayaInput.h"

#include "DeferredRenderer.h"

#include "EmptyGISolution.h"
#include "ThesisSolution.h"

#include "Globals.h"
#include "GFGLoader.h"
#include "Camera.h"
#include "Scene.h"

#include "IEUtility/IEMath.h"

#include "GLFW/glfw3.h"

int main()
{
	Camera mainRenderCamera =
	{
		90.0f,
		0.1f,
		500.0f,
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
			{ 0.0f, 0.0f, 0.0f, static_cast<float>(LightType::DIRECTIONAL)},
			{ 0.0f, -IEMath::CosF(IEMath::ToRadians(17.0f)), -IEMath::SinF(IEMath::ToRadians(17.0f)), 0.0f },
			{ 1.0f, 1.0f, 1.0f, std::numeric_limits<float>::infinity() }
		}
		// TODO:
		// Add Some point lights
	};
	Light cornellLights[] =
	{
		// Area Light
		// White Color
		// Top of the room (center of white rectangle)
		// Covers Entire Room
		{
			{ 0.0f, 183.0f, 0.0f, static_cast<float>(LightType::AREA)},
			{ 0.0f, 1.0f, 0.0f, 0.0f },
			{ 1.0f, 1.0f, 1.0f, 220.0f }
		}
	};
	Scene crySponza(Scene::sponzaFileName, { sponzaLights, 1});
	Scene cornellBox(Scene::cornellboxFileName, {cornellLights, 1});
	scenes.push_back(&crySponza);
	scenes.push_back(&cornellBox);

	// Solutions
	EmptyGISolution emptySolution(deferredRenderer);
	ThesisSolution thesisSolution;
	solutions.push_back(&emptySolution);
	solutions.push_back(&thesisSolution);

	// All Init
	// Render Loop
	while(!mainWindow.WindowClosed())
	{
		// Constantly Check Input Scheme Change
		mainWindow.ChangeInputScheme(*inputSchemes[currentInputScheme % inputSchemes.size()]);

		// Enforce intialization if solution changed
		bool forceInit = false;
		if(oldSolution != currentSolution)
		{
			forceInit = true;
			oldSolution = currentSolution;
		}
			
		SolutionI* solution = solutions[currentSolution % solutions.size()];
		if(!solution->IsCurrentScene(*scenes[currentScene % scenes.size()]) ||
		   forceInit)
		{
			solution->Init(*scenes[currentScene % scenes.size()]);
		}
			
		solution->Frame(mainRenderCamera);
		
		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();
	}
	return 0;
}