#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Globals.h"

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
	glDisable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);	
	glDepthMask(true);
	glColorMask(true, true, true, true);
	currentScene = &s;
	Shader::Unbind(ShaderType::GEOMETRY);
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
	DrawBuffer& dBuffer = currentScene->getDrawBuffer();
	currentScene->getGPUBuffer().Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();

	for(unsigned int i = 0; i < currentScene->DrawCount(); i++)
	{
		dBuffer.BindMaterialForDraw(i);
		dBuffer.getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));
	}
}