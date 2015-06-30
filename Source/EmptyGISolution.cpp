#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DeferredRenderer.h"
#include "Macros.h"


EmptyGISolution::EmptyGISolution(DeferredRenderer& defferedRenderer)
	: currentScene(nullptr)
	, dRenderer(defferedRenderer)
	, bar(nullptr)
{}

bool EmptyGISolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}
void EmptyGISolution::Init(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Bar Creation
	bar = TwNewBar("EmptyGI");
	TwDefine(" EmptyGI refresh=0.01 ");

	// FPS Show
	TwAddVarRO(bar, "fTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame(ms)' help='Frame Time in milliseconds..' ");
}

void EmptyGISolution::Release()
{
	// Release Tweakbar
	if(bar) TwDeleteBar(bar);
}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	dRenderer.Render(*currentScene, mainRenderCamera);
}

void EmptyGISolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}