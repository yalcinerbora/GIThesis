#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DeferredRenderer.h"
#include "Macros.h"
#include "SceneLights.h"
#include "Globals.h"
#include "WindowInput.h"
#include <GLFW/glfw3.h>

EmptyGISolution::EmptyGISolution(const std::string& name, 
								 WindowInput& inputManager,
								 DeferredRenderer& defferedRenderer)
	: name(name)
	, currentScene(nullptr)
	, dRenderer(defferedRenderer)
	, directLighting(true)
	, ambientLighting(true)
	, ambientColor(0.1f, 0.1f, 0.1f)
	, emptyGIBar()
	, scheme(RenderScheme::FINAL)
	, frameTime(0.0)
	, shadowTime(0.0)
	, dPassTime(0.0)
	, gPassTime(0.0)
	, lPassTime(0.0)
	, mergeTime(0.0)
{
	inputManager.AddKeyCallback(GLFW_KEY_KP_ADD, GLFW_RELEASE, &EmptyGISolution::Up, this);
	inputManager.AddKeyCallback(GLFW_KEY_KP_SUBTRACT, GLFW_RELEASE, &EmptyGISolution::Down, this);
	inputManager.AddKeyCallback(GLFW_KEY_KP_MULTIPLY, GLFW_RELEASE, &EmptyGISolution::Next, this);
	inputManager.AddKeyCallback(GLFW_KEY_KP_DIVIDE, GLFW_RELEASE, &EmptyGISolution::Previous, this);
}

bool EmptyGISolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}
void EmptyGISolution::Load(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Attach new Scene's Light Indices
	dRenderer.AttachSceneLightIndices(s);

	// Init GUI
	lightBar = std::move(LightBar());
	lightBar = std::move(LightBar(currentScene->getSceneLights(),
								  directLighting,
								  ambientLighting,
								  ambientColor));

	emptyGIBar = std::move(EmptyGIBar());
	emptyGIBar = std::move(EmptyGIBar(currentScene->getSceneLights(),
									  scheme,
									  frameTime,
									  shadowTime,
									  dPassTime,
									  gPassTime,
									  lPassTime,
									  mergeTime));
}

void EmptyGISolution::Release()
{}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	// Do Deferred Rendering
	bool doTiming = emptyGIBar.DoTiming();
	IEVector3 aColor = ambientLighting ? ambientColor : IEVector3::ZeroVector;
	dRenderer.Render(*currentScene, mainRenderCamera, directLighting, aColor, doTiming);

	// Get Timings
	shadowTime = dRenderer.ShadowMapTime();
	dPassTime = dRenderer.DPassTime();
	gPassTime = dRenderer.GPassTime();
	lPassTime = dRenderer.LPassTime();
	mergeTime = dRenderer.MergeTime();

	if(scheme >= RenderScheme::G_DIFF_ALBEDO &&
	   scheme <= RenderScheme::G_DEPTH)
	{
		dRenderer.ShowGBufferTexture(mainRenderCamera, scheme);
	}
	else if(scheme == RenderScheme::LIGHT_INTENSITY)
	{
		dRenderer.ShowLightIntensity(mainRenderCamera);
	}
	else if(scheme == RenderScheme::SHADOW_MAP)
	{
		dRenderer.ShowShadowMap(mainRenderCamera, *currentScene,
								emptyGIBar.CurrentLight(),
								emptyGIBar.CurrentLevel());
	}
}

void EmptyGISolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}

const std::string& EmptyGISolution::Name() const
{
	return name;
}

void EmptyGISolution::Next()
{
	emptyGIBar.Next();
}

void EmptyGISolution::Previous()
{
	emptyGIBar.Previous();
}

void EmptyGISolution::Up()
{
	emptyGIBar.Up();
}

void EmptyGISolution::Down()
{
	emptyGIBar.Down();
}