#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"

EmptyGISolution::EmptyGISolution()
	: currentScene(nullptr)
	, vertexGBufferWrite(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert")
	, fragmentGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag")
{}


bool EmptyGISolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}
void EmptyGISolution::Init(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	currentScene = &s;
}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	// Start With a VP Set
	// Using a callback is not necessarly true since it may alter some framebuffer's viewport
	// but we have to be sure that it alters main fbo viewport
	glViewport(0, 0,
			   static_cast<GLsizei>(mainRenderCamera.width),
			   static_cast<GLsizei>(mainRenderCamera.height));

	glClear(GL_COLOR_BUFFER_BIT |
			GL_DEPTH_BUFFER_BIT |
			GL_STENCIL_BUFFER_BIT);

	// Camera Transform
	cameraTransform.Update(mainRenderCamera.generateTransform());
	cameraTransform.Bind();

	// Shaders
	vertexGBufferWrite.Bind();
	fragmentGBufferWrite.Bind();

	// DrawCall
	currentScene->Draw();
}