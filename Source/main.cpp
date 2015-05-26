#include <iostream>
#include <GFG/GFGHeader.h>

#include "Window.h"

#include "NoInput.h"
#include "FPSInput.h"
#include "MayaInput.h"

#include "EmptyGISolution.h"
#include "ThesisSolution.h"

#include "Globals.h"
#include "GFGLoader.h"
#include "Camera.h"
#include "Scene.h"

#include "GLFW/glfw3.h"

int main()
{
	Camera mainRenderCamera =
	{
		60.0f,
		0.1f,
		10000.0f,
		1280,
		720,
		{ 150.0f, -145.0f, 0.3f },
		{ -0.15f, -37.0f, 3.4f },
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

	// Scenes
	Scene crySponza(Scene::sponzaFileName, {{}, 0});
	Scene cornellBox(Scene::cornellboxFileName, {{}, 0});
	scenes.push_back(&crySponza);
	scenes.push_back(&cornellBox);

	// Solutions
	EmptyGISolution emptySolution;
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