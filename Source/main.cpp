#include <iostream>
#include <GFG/GFGHeader.h>

#include "Window.h"

#include "NoInput.h"
#include "FPSInput.h"
#include "MayaInput.h"

#include "DeferredRenderer.h"
#include "BatchFunctors.h"

#include "EmptyGISolution.h"
#include "ThesisSolution.h"

#include "Globals.h"
#include "Camera.h"
#include "Macros.h"
#include "CudaInit.h"

#include "ThesisScenes.h"

#include "IEUtility/IEMath.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IETimer.h"
#include "IEUtility/IERandom.h"
#include <GLFW/glfw3.h>

int main()
{
	// Cuda Init
	CudaInit::InitCuda();

	Camera mainRenderCamera =
	{
		75.0f,
		0.15f,
		650.0f,
		1280,
		720,
		IEVector3(-180.0f, 145.0f, 0.3f),
		IEVector3::ZeroVector,
		IEVector3::YAxis
	};

	std::vector<SolutionI*> solutions;
	std::vector<SceneI*> scenes;
	std::vector<CameraInputI*> cameraInputSchemes;

	// Input Schemes
	NoInput nullInput;
	MayaInput mayaInput(0.005, 0.1, 0.2);
	FPSInput fpsInput(0.005, 4.30, 0.25);
	cameraInputSchemes.push_back(&nullInput);
	cameraInputSchemes.push_back(&mayaInput);
	cameraInputSchemes.push_back(&fpsInput);

	// Acutal Input
	WindowInput inputManager(mainRenderCamera,
							 cameraInputSchemes,
							 solutions,
							 scenes);
	// Window Init
	WindowProperties winProps
	{
		1280, 720,
		WindowScreenType::WINDOWED
	};
	Window mainWindow("GI Thesis", inputManager, winProps);

	// GUI Help
	TwDefine(" GLOBAL iconpos=tl ");
	TwDefine(" GLOBAL help='GI Implementation using voxels.\n"
			 "\tUse NumPad 7,8 to change between solutions.\n"
			 "\tUse NumPad 4,6 to change between scenes.\n"
			 "\tUse NumPad 1,3 to change between camera input schemes.\n"
			 "\t\t Input Scheme#1 : No Input.\n"
			 "\t\t Input Scheme#2 : Maya Input. (MouseBTN1 to rotate around COI. Mouse BTN3 to translate COI)\n"
			 "\t\t Input Scheme#3 : FPS Input. (WASD to move MouseBTN1 to look around)\n"
			 "' ");

	// DeferredRenderer
	DeferredRenderer deferredRenderer;

	// Scenes
	IEVector3 lightDir = IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * -84.266),
									  IEVector3::XAxis).ApplyRotation(-IEVector3::ZAxis);
	std::vector<Light> sponzaLights =
	{
		// Directional Light
		// White Color
		// 1-2 PM Sunlight direction (if you consider lionhead(window) is at north)
		{
			{ 0.0f, 0.0f, 0.0f, static_cast<float>(LightType::DIRECTIONAL) },
			{lightDir.getX(), lightDir.getY(), lightDir.getZ(), std::numeric_limits<float>::infinity()},
			//{ 0.0f, -IEMath::CosF(IEMath::ToRadians(9.5f)), -IEMath::SinF(IEMath::ToRadians(9.5f)), std::numeric_limits<float>::infinity()},
			IEVector4(1.0f, 1.0f, 1.0f, 4.2f)
		},
		// TODO: Point lights have bugs, direct depth write do not work properly
		////Point Lights
		////Various Colors color effecting radius 60 units
		//{
		//	{ 212.6f, 50.8f, -85.3f, static_cast<float>(LightType::POINT) },
		//	{ 0.0f, 0.0f, 0.0f, 120.0f },
		//	IEVector4(1.0f, 1.0f, 1.0f, 3000.0f)
		//},
		//{
		//	{ -116.8f, 27.5f, 17.0f, static_cast<float>(LightType::POINT) },
		//	{ 0.0f, 0.0f, 0.0f, 120.0f },
		//	IEVector4(1.0f, 1.0f, 1.0f, 3000.0f)
		//},
		//{
		//	{ 92.2f, 25.9f, 16.2f, static_cast<float>(LightType::POINT) },
		//	{ 0.0f, 0.0f, 0.0f, 120.0f },
		//	IEVector4(1.0f, 1.0f, 1.0f, 3000.0f)
		//}
	};
	std::vector<Light> sibernikLights =
	{
		// Directional Light
		// White Color
		// Diretly Comes from Front Door
		{
			{0.0f, 0.0f, 0.0f, static_cast<float>(LightType::DIRECTIONAL)},
			{
				std::sin(static_cast<float>(IEMathConstants::DegToRadCoef * 45.0)),
				-std::cos(static_cast<float>(IEMathConstants::DegToRadCoef * 45.0)),
				0.0f,
				std::numeric_limits<float>::infinity()
			},
			IEVector4(1.0f, 1.0f, 1.0f, 4.2f)
		},
		////Point Lights
		//{
		//	{-80.0f, 100.0f, 0.0f, static_cast<float>(LightType::POINT)},
		//	{0.0f, 0.0f, 0.0f, 1000.0f},
		//	IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		//},
		//{
		//	{0.0f, 100.0f, 0.0f, static_cast<float>(LightType::POINT)},
		//	{0.0f, 0.0f, 0.0f, 1000.0f},
		//	IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		//},
		//{
		//	{80.0f, 100.0f, 0.0f, static_cast<float>(LightType::POINT)},
		//	{0.0f, 0.0f, 0.0f, 1000.0f},
		//	IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		//}
	};
	std::vector<Light> cornellLights =
	{
		// Point Light
		// White Color
		// Top of the room (center of white rectangle)
		// Covers Entire Room
		//{
		//	{ 0.0f, 160.0f, 0.0f, static_cast<float>(LightType::POINT)},
		//	{ 0.0f, -1.0f, 0.0f, 3300.0f},
		//	IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		//}
		{
			{0.0f, 0.0f, 0.0f, static_cast<float>(LightType::DIRECTIONAL)},
			{0.56f, -0.69f, 0.46f, std::numeric_limits<float>::infinity()},
			IEVector4(1.0f, 1.0f, 1.0f, 4.2f)
		}
	};

	// GFG File Names
	const std::string SponzaFileName = "sponza.gfg";
	const std::string SponzaDynamicFileName = "sponzaDynamic.gfg";

	const std::string NyraFileName = "nyra.gfg";
	const std::string CornellboxFileName = "cornell.gfg";
	const std::string SibernikFileName = "sibernik.gfg";
	const std::string DynamicFileName = "dynamicScene.gfg";

	// Scene Files
	// Sponza Atrium
	std::vector<std::string> sponzaRigid =
	{
		"sponza.gfg",
		"sponzaDynamic.gfg",
	};
	std::vector<std::string> sponzaSkeletal =
	{
		"nyra.gfg"
	};
	SponzaScene sponza("Sponza Atrium", sponzaRigid, sponzaSkeletal, sponzaLights);
	scenes.push_back(&sponza);
	//// Cornell
	//std::vector<std::string> cornellRigid =
	//{
	//	"cornell.gfg"
	//};
	//std::vector<std::string> cornellSkeletal = {};
	//CornellScene cornell("Cornell Box", cornellRigid, cornellSkeletal, cornellLights);
	//scenes.push_back(&cornell);
	//// Sibernik Cathedral
	//std::vector<std::string> sibernikRigid =
	//{
	//	"sibernik.gfg"
	//};
	//std::vector<std::string> sibernikSkeletal = {};
	//ConstantScene sibernik("Sibernik Cathedral", sibernikRigid, sibernikSkeletal, sibernikLights);
	//scenes.push_back(&sibernik);
	//// Nyra Single
	//std::vector<std::string> nyraStatic = {};
	//ConstantScene nyraSingle("Nyra Single", nyraStatic, sponzaSkeletal, sponzaLights);
	//scenes.push_back(&nyraSingle);
	//// Dynamic Scene
	//std::vector<std::string> dynamicRigid =
	//{
	//	"dynamicScene.gfg"
	//};
	//static constexpr int repeatCount = 256;
	//std::vector<std::string> dynamicSkeletal
	//{
	//	"nyra.gfg"
	//};
	//DynoScene dynamic("Dynamic Scene", dynamicRigid, dynamicSkeletal, sponzaLights, repeatCount);
	//scenes.push_back(&dynamic);

	// Solutio ns
	EmptyGISolution emptySolution(inputManager, deferredRenderer, "No GI");
	ThesisSolution thesisSolution(5,		// Dense Level
								  5,		// Dense Count
								  3,		// Cascade Count
								  8,		// Base Level
								  1.00f,	// Base Span
								  true,		// Use cache
								  inputManager, deferredRenderer, "Thesis GI");
	solutions.push_back(&emptySolution);
	solutions.push_back(&thesisSolution);
	// All Done
	// Initialize First Scene and Solution
	inputManager.Initialize();

	// FPS Timer
	IETimer t;
	t.Start();

	// Pre-defined cameras for paper
	// Relfection test, dynamic scene
	//IEVector3 camPos = IEVector3(-223.717896f, 69.1969299f, -131.224045f);
	//IEVector3 camLook = IEVector3(-73.4765625f, 40.8114090f, -304.566162f);
	// Sample test, walking nyra
	//IEVector3 camPos = IEVector3(3.94091797f, 40.0365143f, 63.8204422f);
	//IEVector3 camLook = IEVector3(2.62701273f, -17.8482475f, -159.956665f);
	// Dyno test
	//IEVector3 camPos = IEVector3(-18.1055393f, 103.491150f, 318.622131f);
	//IEVector3 camLook = IEVector3(-20.4996319f, 33.3934479f, 98.3792267f);
	// Center Dyno
	IEVector3 camPos = IEVector3(-44.2577820f, 101.216194f, 26.8820076f);
	IEVector3 camLook = IEVector3(-41.6408730f, -63.8801727f, -73.3516541f);

	mainRenderCamera.centerOfInterest = camLook;
	mainRenderCamera.pos = camPos;

	// Render Loop
	while(!mainWindow.WindowClosed())
	{
        double elapsedTime = t.ElapsedS();

		auto solution = inputManager.Solution();
		auto scene = inputManager.Scene();

		// Moving light test
		constexpr double speed = 2.5f * IEMathConstants::DegToRadCoef; // Radians / sec
		IEVector3 direction = scene->getSceneLights().getLightDir(0);
		direction = IEMatrix4x4::Rotate(static_cast<float>(speed * elapsedTime),
										-IEVector3::XAxis) * direction;
		//scene->getSceneLights().ChangeLightDir(0, direction);

		// Update Scene Render Frame
		scene->Update(elapsedTime);
		solution->Frame(mainRenderCamera);

		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();

		t.Lap();
		solution->SetFPS(t.ElapsedMilliS());
	}
	return 0;
}
