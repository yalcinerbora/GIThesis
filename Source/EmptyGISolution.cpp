#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DeferredRenderer.h"


EmptyGISolution::EmptyGISolution(DeferredRenderer& defferedRenderer)
	: currentScene(nullptr)
	, dRenderer(defferedRenderer)
{}

bool EmptyGISolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}
void EmptyGISolution::Init(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;
}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	dRenderer.Render(*currentScene, mainRenderCamera);
}