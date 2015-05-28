#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DrawBuffer.h"
#include "Globals.h"
#include "SceneLights.h"
#include "IEUtility/IEMath.h"

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
	IEVector3 worldFrustumMax;
	IEVector3 worldFrustumMin;

	// Calculate Frustum Parameters from Render Camera
	float tanHalfFovX = IEMath::TanF(IEMath::ToRadians(mainRenderCamera.fovX * 0.5f));
	IEVector3 camDir = (mainRenderCamera.centerOfInterest - mainRenderCamera.pos).NormalizeSelf();
	IEVector3 right = camDir.CrossProduct(mainRenderCamera.up);

	float nearWidth = 2 * mainRenderCamera.near * tanHalfFovX;
	float nearHeight = nearWidth / (mainRenderCamera.width / mainRenderCamera.height);
	float farWidth = 2 * mainRenderCamera.far * tanHalfFovX;
	float farHeight = farWidth / (mainRenderCamera.width / mainRenderCamera.height);

	// Plane Center Points
	IEVector3 planeCenterNear = mainRenderCamera.pos + camDir * mainRenderCamera.near;
	IEVector3 planeCenterFar = mainRenderCamera.pos + camDir * mainRenderCamera.far;

	// Frustum Points
	//IEVector3 nearTopLeft = planeCenterNear + (mainRenderCamera.up * nearHeight * 0.5f) - (right * nearWidth * 0.5f);
	//IEVector3 nearTopRight = planeCenterNear + (mainRenderCamera.up * nearHeight * 0.5f) + (right * nearWidth * 0.5f);
	//IEVector3 nearBottomLeft = planeCenterNear - (mainRenderCamera.up * nearHeight * 0.5f) - (right * nearWidth * 0.5f);
	//IEVector3 nearBottomRight = planeCenterNear - (mainRenderCamera.up * nearHeight * 0.5f) + (right * nearWidth * 0.5f);

	//IEVector3 farTopLeft = planeCenterFar + (mainRenderCamera.up * farHeight * 0.5f) - (right * farWidth * 0.5f);
	IEVector3 farTopRight = planeCenterFar + (mainRenderCamera.up * farHeight * 0.5f) + (right * farWidth * 0.5f);
	IEVector3 farBottomLeft = planeCenterFar - (mainRenderCamera.up * farHeight * 0.5f) - (right * farWidth * 0.5f);
	//IEVector3 farBottomRight = planeCenterFar - (mainRenderCamera.up * farHeight * 0.5f) + (right * farWidth * 0.5f);
	
	// Make view frustum rectangular
	IEVector3 nearBottomLeftRect = farBottomLeft - (mainRenderCamera.far - mainRenderCamera.near) * camDir;
	worldFrustumMax = farTopRight;
	worldFrustumMin = nearBottomLeftRect;

	// Actual Render Call for Shadow Maps
	currentScene->getSceneLights().GenerateShadowMaps(currentScene->getDrawBuffer(),
													  currentScene->getGPUBuffer(),
													  cameraTransform,
													  static_cast<unsigned int>(currentScene->DrawCount()),
													  worldFrustumMin,
													  worldFrustumMax);

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