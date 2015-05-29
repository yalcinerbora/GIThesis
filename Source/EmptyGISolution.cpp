#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Globals.h"
#include "SceneLights.h"
#include "IEUtility/IEMath.h"
#include "RectPrism.h"

EmptyGISolution::EmptyGISolution()
	: currentScene(nullptr)
	, gBuffer(1280, 720)
	, vertexGBufferWrite(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert")
	, fragmentGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag")
	, vertLightPass(ShaderType::VERTEX, "Shaders/LightPass.vert")
	, fragLightPass(ShaderType::FRAGMENT, "Shaders/LightPass.frag")
	, vertPPGeneric(ShaderType::VERTEX, "Shaders/PProcessGeneric.vert")
	, fragLightApply(ShaderType::FRAGMENT, "Shaders/PPLightPresent.frag")
{}

bool EmptyGISolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}
void EmptyGISolution::Init(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;
	Shader::Unbind(ShaderType::GEOMETRY);
}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	// Shadow Map Generation
	// Calculate Frustum Parameters from Render Camera
	float tanHalfFovX = IEMath::TanF(IEMath::ToRadians(mainRenderCamera.fovX * 0.5f));
	float aspectRatio = mainRenderCamera.width / mainRenderCamera.height;
	IEVector3 camDir = (mainRenderCamera.centerOfInterest - mainRenderCamera.pos).NormalizeSelf();
	IEVector3 right = camDir.CrossProduct(mainRenderCamera.up);

	float farHalfWidth =  mainRenderCamera.far * tanHalfFovX;
	float farHalfHeight = farHalfWidth / aspectRatio;

	// Plane Center Points
	IEVector3 planeCenterFar = mainRenderCamera.pos + camDir * mainRenderCamera.far;
	//IEVector3 farTopLeft = planeCenterFar + (mainRenderCamera.up * farHeight * 0.5f) - (right * farWidth * 0.5f);
	IEVector3 farTopRight = planeCenterFar + (mainRenderCamera.up * farHalfHeight) + (right * farHalfWidth);
	IEVector3 farBottomLeft = planeCenterFar - (mainRenderCamera.up * farHalfHeight) - (right * farHalfWidth);
	IEVector3 farBottomRight = planeCenterFar - (mainRenderCamera.up * farHalfHeight) + (right * farHalfWidth);

	// MRectangular Prism View Frustum Coords in World Space
	const IEVector3 span[3] = 
	{
		farTopRight - farBottomRight,
		-mainRenderCamera.far * camDir,
		farBottomLeft - farBottomRight
	};
	RectPrism viewFrustum(span, farBottomRight);

	// Actual Render Call for Shadow Maps
	currentScene->getSceneLights().GenerateShadowMaps(currentScene->getDrawBuffer(),
													  currentScene->getGPUBuffer(),
													  cameraTransform,
													  static_cast<unsigned int>(currentScene->DrawCount()),
													  viewFrustum);

	// Actual Render
	// Start With a VP Set
	// Using a callback is not necessarly true since it may alter some framebuffer's viewport
	// but we have to be sure that it alters main fbo viewport
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(mainRenderCamera.width),
			   static_cast<GLsizei>(mainRenderCamera.height));

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDepthMask(true);
	glColorMask(true, true, true, true);
	glClear(GL_COLOR_BUFFER_BIT |
			GL_DEPTH_BUFFER_BIT |
			GL_STENCIL_BUFFER_BIT);

	// Camera Transform
	cameraTransform.Update(mainRenderCamera.generateTransform());
	cameraTransform.Bind();

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
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