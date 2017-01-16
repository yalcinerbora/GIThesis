#include <iostream>
#include <GFG/GFGHeader.h>
#include <AntTweakBar.h>

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
#include "Scene.h"
#include "Macros.h"
#include "CudaInit.h"

#include "MeshBatchNyra.h"
#include "MeshBatchCornell.h"
#include "MeshBatchSponza.h"
#include "MeshBatchCube.h"
#include "MeshBatchOscillate.h"
#include "MeshBatchDyno.h"
#include "MeshBatchMulti.h"

#include "IEUtility/IEMath.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IETimer.h"
#include "IEUtility/IERandom.h"
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
		//WindowScreenType::FULLSCREEN
	};
	Window mainWindow(nullInput,
					  winProps);

	// DeferredRenderer
	DeferredRenderer deferredRenderer;

	// Scenes
	IEVector3 lightDir = IEQuaternion(IEMath::ToRadians(-84.266f), IEVector3::Xaxis).ApplyRotation(-IEVector3::Zaxis);
	Light sponzaLights[] =
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
	Light sibernikLights[] =
	{
		// Directional Light
		// White Color
		// Diretly Comes from Front Door
		{
			{0.0f, 0.0f, 0.0f, static_cast<float>(LightType::DIRECTIONAL)},
			{IEMath::SinF(IEMath::ToRadians(45.0f)), -IEMath::CosF(IEMath::ToRadians(45.0f)), 0.0f, std::numeric_limits<float>::infinity()},
			IEVector4(1.0f, 1.0f, 1.0f, 4.2f)
		},
		//Point Lights
		{
			{-80.0f, 100.0f, 0.0f, static_cast<float>(LightType::POINT)},
			{0.0f, 0.0f, 0.0f, 1000.0f},
			IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		},
		{
			{0.0f, 100.0f, 0.0f, static_cast<float>(LightType::POINT)},
			{0.0f, 0.0f, 0.0f, 1000.0f},
			IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		},
		{
			{80.0f, 100.0f, 0.0f, static_cast<float>(LightType::POINT)},
			{0.0f, 0.0f, 0.0f, 1000.0f},
			IEVector4(1.0f, 1.0f, 1.0f, 11000.0f)
		}
	};
	Light cornellLights[] =
	{
		// Area Light
		// White Color
		// Top of the room (center of white rectangle)
		// Square light
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

	//// Sponza Scene
	//MeshBatch crySponzaStatic(MeshBatch::sponzaFileName,
	//						  ThesisSolution::CascadeSpan,
	//						  false);
	//MeshBatchSponza crySponzaDynamic(MeshBatchSponza::sponzaDynamicFileName,
	//								  ThesisSolution::CascadeSpan);
	//MeshBatchNyra nyraBatch(MeshBatchSkeletal::nyraFileName,
	//						ThesisSolution::CascadeSpan);
	//nyraBatch.AnimationParams(0.0f, 0.6f, AnimationType::REPEAT);
 //   MeshBatchI* sponzaBatches[] = {&crySponzaStatic, &crySponzaDynamic, &nyraBatch};
 //   Scene crySponza(Array32<MeshBatchI*>{sponzaBatches, 3},
 //                   Array32<Light>{sponzaLights, 1},
 //                   Scene::bigSizes);
 //   scenes.push_back(&crySponza);

	//// Floor Scene
	//class MeshBatchFloor : public MeshBatch
	//{
	//	private:
	//	protected:
	//	public:
	//	// Constructors & Destructor
	//	MeshBatchFloor(const char* sceneFileName, float minVoxSpan)
	//		:MeshBatch(sceneFileName, minVoxSpan, false)
	//	{}

	//	// Interface
	//	void Update(double elapsedS) override
	//	{
	//		static const float translateSpeed = 0.3f;
	//		std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();
	//		BatchFunctors::ApplyTranslation translator(mtBuff);
	//		translator(1, elapsedS, IEVector3::Zaxis * translateSpeed);
	//		batchDrawParams.getModelTransformBuffer().SendData();
	//	}
	//	VoxelObjectType MeshType() const override { return VoxelObjectType::DYNAMIC; }
	//} floorBatch("testFloor.gfg", ThesisSolution::CascadeSpan);
	////} floorBatch("testBox.gfg", ThesisSolution::CascadeSpan);
	//MeshBatchI* floorBatches[] = {&floorBatch};
	//Scene floorScene(Array32<MeshBatchI*>{floorBatches, 1},
	//				Array32<Light>{sponzaLights, 1},
	//				Scene::sponzaSceneLevelSizes);
	//scenes.push_back(&floorScene);

	// Nyra Comparison Scene
	MeshBatchSkeletal nyraBatch(MeshBatchSkeletal::nyraFileName,
								ThesisSolution::CascadeSpan);
	MeshBatchI* nyraBatches[] = {&nyraBatch};
	Scene nyraCompare(Array32<MeshBatchI*>{nyraBatches, 1},
					  Array32<Light>{sponzaLights, 1},
					  Scene::bigSizes);
	nyraBatch.AnimationParams(0.0f, 0.6f, AnimationType::REPEAT);
	scenes.push_back(&nyraCompare);

	//// Cornell Box Scene
	//MeshBatch cornellStatic(MeshBatch::cornellboxFileName,
	//						ThesisSolution::CascadeSpan,
	//						false);
	//MeshBatchCornell cornellDynamic(MeshBatchCornell::cornellDynamicFileName,
	//								ThesisSolution::CascadeSpan);
 //   MeshBatchI* cornellBatches[] = {&cornellStatic, &cornellDynamic};
 //   Scene cornellBox(Array32<MeshBatchI*>{cornellBatches, 2},
 //                    Array32<Light>{cornellLights, 1},
 //                    Scene::cornellSceneLevelSizes);
 //   scenes.push_back(&cornellBox);

	//// Sibernik Scene
	//MeshBatch sibernikStatic(MeshBatch::sibernikFileName,
	//						 ThesisSolution::CascadeSpan,
	//						 false);
 //   MeshBatchI* sibernikBatches[] = {&sibernikStatic};
 //   Scene sibernik(Array32<MeshBatchI*>{sibernikBatches, 1},
 //                  Array32<Light>{sibernikLights, 1},
 //                  Scene::sibernikSceneLevelSizes);
 //   scenes.push_back(&sibernik);
	
    //// Dynamic Scene
    //MeshBatchDyno dynamicBatch(MeshBatch::dynamicFileName,
    //                           ThesisSolution::CascadeSpan);
    //MeshBatchMultiSkel nyraJump(MeshBatchMultiSkel::nyraName,
    //                           ThesisSolution::CascadeSpan,
    //                            -150.0f, -150.0f, 30.0f, 300.0f,
    //                            128);
    //nyraJump.AnimationParams(0.0f, 0.3f, AnimationType::REPEAT);
    //MeshBatchI* dynamicScalingBatches[] = {&dynamicBatch, &nyraJump};
    //Scene dynamicScene(Array32<MeshBatchI*>{dynamicScalingBatches, 2},
    //                   Array32<Light>{sponzaLights, 1},
    //                   Scene::dynamicSceneLevelSizes);
    //scenes.push_back(&dynamicScene);
	
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

	nullInput.AddKeyCallback(GLFW_KEY_KP_DIVIDE, GLFW_RELEASE, &ThesisSolution::TraceDecrement, &thesisSolution);
	nullInput.AddKeyCallback(GLFW_KEY_KP_MULTIPLY, GLFW_RELEASE, &ThesisSolution::TraceIncrement, &thesisSolution);
	mayaInput.AddKeyCallback(GLFW_KEY_KP_DIVIDE, GLFW_RELEASE, &ThesisSolution::TraceDecrement, &thesisSolution);
	mayaInput.AddKeyCallback(GLFW_KEY_KP_MULTIPLY, GLFW_RELEASE, &ThesisSolution::TraceIncrement, &thesisSolution);
	fpsInput.AddKeyCallback(GLFW_KEY_KP_DIVIDE, GLFW_RELEASE, &ThesisSolution::TraceDecrement, &thesisSolution);
	fpsInput.AddKeyCallback(GLFW_KEY_KP_MULTIPLY, GLFW_RELEASE, &ThesisSolution::TraceIncrement, &thesisSolution);

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

	// Render Loop
	while(!mainWindow.WindowClosed())
	{
        double elapsedTime = t.ElapsedS();

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

        // Toggle Time
        if(inputSchemes[currentInputScheme % inputSchemes.size()]->MoveLight())
            elapsedTime = 0.0;

        // Toggle Dir Movement
        if(inputSchemes[currentInputScheme % inputSchemes.size()]->MoveLight())
            scenes[currentScene % scenes.size()]->getSceneLights().ChangeLightDir(0, IEQuaternion(IEMath::ToRadians(elapsedTime * 5.0f),
                    -IEVector3::Xaxis).ApplyRotation(scenes[currentScene % scenes.size()]->getSceneLights().GetLightDir(0)));

		// Render frame
		scenes[currentScene % scenes.size()]->Update(elapsedTime);
		solution->Frame(mainRenderCamera);
		
		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();

		t.Lap();
		solution->SetFPS(t.ElapsedMilliS());
	}

	//for(MeshBatchI*& batch : nyraVector)
	//{
	//	delete batch;
	//	batch = nullptr;
	//}
	return 0;
}
