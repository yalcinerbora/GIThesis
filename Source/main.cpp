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
#include "MeshBatchDynamic.h"
#include "BatchUpdates.h"
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

	IEVector3 camPos = IEVector3(-180.0f, 145.0f, 0.3f);
	Camera mainRenderCamera =
	{
		90.0f,
		0.15f,
		600.0f,
		1280,
		720,
		camPos,
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

	// Sponza Scene
	MeshBatchStatic crySponzaStatic(MeshBatchStatic::sponzaFileName,
									ThesisSolution::CascadeSpan / 0.19f,
									{MeshBatchStatic::sponzaVoxelSizes, GI_CASCADE_COUNT});
	MeshBatchDynamic crySponzaDynamic(MeshBatchDynamic::sponzaDynamicFileName,
									  ThesisSolution::CascadeSpan,
									  {MeshBatchDynamic::sponzaDynamicVoxelSizes, GI_CASCADE_COUNT},
									  BatchUpdates::SponzaUpdate);

	// Cornell Box Scene
	MeshBatchStatic cornellStatic(MeshBatchStatic::cornellboxFileName,
								  ThesisSolution::CascadeSpan,
								  {MeshBatchDynamic::cornellVoxelSizes, GI_CASCADE_COUNT});
	MeshBatchDynamic cornellDynamic(MeshBatchDynamic::cornellDynamicFileName,
									ThesisSolution::CascadeSpan,
									{MeshBatchDynamic::cornellDynamicVoxelSizes, GI_CASCADE_COUNT},
									BatchUpdates::CornellUpdate);

	// Cube Scene
	MeshBatchDynamic cubeRotateBatch(MeshBatchDynamic::rotatingCubeFileName,
									 ThesisSolution::CascadeSpan,
									 {MeshBatchDynamic::rotatingCubeVoxelSizes, GI_CASCADE_COUNT},
									 BatchUpdates::CubeUpdate);

	// Scene Interfaces
	MeshBatchI* sponzaBatches[] = {&crySponzaStatic, &crySponzaDynamic};
	Scene crySponza(Array32<MeshBatchI*>{sponzaBatches, 1},
					Array32<Light>{sponzaLights, 4},
					Scene::sponzaSceneTotalSize,
					Scene::sponzaSceneLevelSizes);

	MeshBatchI* cornellBatches[] = {&cornellStatic, &cornellDynamic};
	Scene cornellBox(Array32<MeshBatchI*>{cornellBatches, 2},
					 Array32<Light>{cornellLights, 1},
					 Scene::cornellSceneTotalSize,
					 Scene::cornellSceneLevelSizes);

	MeshBatchI* cubeBatches[] = {&cubeRotateBatch};
	Scene cubeRotate(Array32<MeshBatchI*>{cubeBatches, 1},
					 Array32<Light>{sponzaLights, 1},
					 Scene::cubeSceneTotalSize,
					 Scene::cubeSceneLevelSizes);
	scenes.push_back(&crySponza);
	scenes.push_back(&cornellBox);
	scenes.push_back(&cubeRotate);

	// Solutions
	EmptyGISolution emptySolution(deferredRenderer);
	ThesisSolution thesisSolution(deferredRenderer, mainRenderCamera.pos);
	solutions.push_back(&emptySolution);
	solutions.push_back(&thesisSolution);

	// Window Callbacks (Thesis Solution Stuff)
	nullInput.AddKeyCallback(GLFW_KEY_KP_ADD, GLFW_RELEASE, &ThesisSolution::LevelIncrement, &thesisSolution);
	nullInput.AddKeyCallback(GLFW_KEY_KP_SUBTRACT, GLFW_RELEASE, &ThesisSolution::LevelDecrement, &thesisSolution);
	mayaInput.AddKeyCallback(GLFW_KEY_KP_ADD, GLFW_RELEASE, &ThesisSolution::LevelIncrement, &thesisSolution);
	mayaInput.AddKeyCallback(GLFW_KEY_KP_SUBTRACT, GLFW_RELEASE, &ThesisSolution::LevelDecrement, &thesisSolution);
	fpsInput.AddKeyCallback(GLFW_KEY_KP_ADD, GLFW_RELEASE, &ThesisSolution::LevelIncrement, &thesisSolution);
	fpsInput.AddKeyCallback(GLFW_KEY_KP_SUBTRACT, GLFW_RELEASE, &ThesisSolution::LevelDecrement, &thesisSolution);

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

		// Render frame
		scenes[currentScene % scenes.size()]->Update(t.ElapsedS());
		solution->Frame(mainRenderCamera);
		
		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();

		t.Lap();
		solution->SetFPS(t.ElapsedMilliS());
	}
	return 0;
}