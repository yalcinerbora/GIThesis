#include "DeferredRenderer.h"
#include "Scene.h"
#include "Globals.h"
#include "IEUtility/IEMath.h"
#include "Camera.h"
#include "RectPrism.h"

const GLsizei DeferredRenderer::gBuffWidth = 1920;
const GLsizei DeferredRenderer::gBuffHeight = 1080;

const float DeferredRenderer::postProcessTriData[6] =
{
	3.0f, -1.0f,
	-1.0f, 3.0f,
	-1.0f, -1.0f
};

DeferredRenderer::DeferredRenderer()
	: gBuffer(gBuffWidth, gBuffHeight)
	, vertexGBufferWrite(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert")
	, fragmentGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag")
	, vertDPass(ShaderType::VERTEX, "Shaders/DPass.vert")
	, vertLightPass(ShaderType::VERTEX, "Shaders/LightPass.vert")
	, fragLightPass(ShaderType::FRAGMENT, "Shaders/LightPass.frag")
	, vertPPGeneric(ShaderType::VERTEX, "Shaders/PProcessGeneric.vert")
	, fragLightApply(ShaderType::FRAGMENT, "Shaders/PPLightPresent.frag")
	, fragShadowMap(ShaderType::FRAGMENT, "Shaders/ShadowMap.frag")
	, vertShadowMap(ShaderType::VERTEX, "Shaders/ShadowMap.vert")
	, geomAreaShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapA.geom")
	, geomDirShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapD.geom")
	, geomPointShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapP.geom")
	, lightIntensityTex(0)
	, lightIntensityFBO(0)
	, invFrameTransform(1)
{
	invFrameTransform.AddData({IEMatrix4x4::IdentityMatrix,
							   IEVector4::ZeroVector,
							   {0, 0, 0, 0},
							   IEVector4::ZeroVector});
	// Light Intensity Tex
	glGenTextures(1, &lightIntensityTex);
	glGenFramebuffers(1, &lightIntensityFBO);

	glBindTexture(GL_TEXTURE_2D, lightIntensityTex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, gBuffWidth, gBuffHeight);

	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, lightIntensityTex, 0);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, gBuffer.getDepthGL(), 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	// PostProcess VAO
	glGenBuffers(1, &postProcessTriBuffer);
	glGenVertexArrays(1, &postProcessTriVao);

	glBindBuffer(GL_COPY_WRITE_BUFFER, postProcessTriBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, sizeof(float) * 6, postProcessTriData, GL_STATIC_DRAW);

	glBindVertexArray(postProcessTriVao);
	glBindVertexBuffer(0, postProcessTriBuffer, 0, sizeof(float) * 2);
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS, 2, GL_FLOAT, false, 0);
	glVertexAttribBinding(IN_POS, 0);

	glGenSamplers(1, &flatSampler);
	glGenSamplers(1, &linearSampler);
	glGenSamplers(1, &shadowMapSampler);

	glSamplerParameteri(flatSampler, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glSamplerParameteri(flatSampler, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	glSamplerParameteri(linearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(linearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_COMPARE_FUNC, GL_GREATER);
}

DeferredRenderer::~DeferredRenderer()
{
	glDeleteTextures(1, &lightIntensityTex);
	glDeleteFramebuffers(1, &lightIntensityFBO);
	glDeleteBuffers(1, &postProcessTriBuffer);
	glDeleteVertexArrays(1, &postProcessTriVao);
	glDeleteSamplers(1, &flatSampler);
	glDeleteSamplers(1, &linearSampler);
}

GBuffer& DeferredRenderer::GetGBuffer()
{
	return gBuffer;
}

void DeferredRenderer::GenerateShadowMaps(SceneI& scene,
										  const Camera&,
										  const RectPrism& viewFrustum)
{
	fragShadowMap.Bind();
	vertShadowMap.Bind();

	// State
	glColorMask(false, false, false, false);
	glDepthMask(true);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_CULL_FACE);
	glViewport(0, 0, SceneLights::shadowMapW, SceneLights::shadowMapH);

	scene.getGPUBuffer().Bind();
	cameraTransform.Bind();
	scene.getDrawBuffer().getDrawParamBuffer().BindAsDrawIndirectBuffer();
	scene.getSceneLights().viewMatrices.BindAsUniformBuffer(U_SHADOW_VIEW);

	// Render From Dir of the light	with proper view params
	IEMatrix4x4 viewTransform = IEMatrix4x4::IdentityMatrix;
	IEMatrix4x4 projection;
	for(int i = 0; i < scene.getSceneLights().lightsGPU.CPUData().size(); i++)
	{
		const Light& currentLight = scene.getSceneLights().lightsGPU.CPUData()[i];
		// Determine light type
		LightType t = static_cast<LightType>(static_cast<uint32_t>(currentLight.position.getW()));
		switch(t)
		{
			case LightType::POINT:
			{
				// Render to Cubemap
				geomPointShadowMap.Bind();

				// Each Side will have 90 degree FOV
				// Geom shader will render for each layer
				for(unsigned int i = 0; i < 6; i++)
				{
					scene.getSceneLights().viewMatrices.CPUData()[i] = 
						IEMatrix4x4::LookAt(currentLight.position,
										    currentLight.position + SceneLights::pLightDir[i],
											SceneLights::pLightUp[i]);
				}
				scene.getSceneLights().viewMatrices.SendData();
				projection = IEMatrix4x4::Perspective(90.0f, 1.0f,
													  0.1f, currentLight.color.getW());
				break;
			}
			case LightType::DIRECTIONAL:
			{
				// Render to one sheet
				geomDirShadowMap.Bind();

				// Camera Direction should be
				viewTransform = IEMatrix4x4::LookAt(currentLight.position,
													currentLight.position + currentLight.direction,
													IEVector3::Yaxis);

				// Span area on viewSpace coordiantes
				RectPrism transRect = viewFrustum.Transform(viewTransform);
				IEVector3 aabbFrustumMin, aabbFrustumMax;
				transRect.toAABB(aabbFrustumMin, aabbFrustumMax);
				projection = IEMatrix4x4::Ortogonal(aabbFrustumMin.getX(), aabbFrustumMax.getX(),
													aabbFrustumMax.getY(), aabbFrustumMin.getY(),
													-500.0f, 500.0f);
				break;
			}
			case LightType::AREA:
			{
				// Render to cube but only 5 sides (6th side is not illuminated)
				geomAreaShadowMap.Bind();
				// we'll use 5 sides but each will comply different ares that a point light
				for(unsigned int i = 0; i < 6; i++)
				{
					scene.getSceneLights().viewMatrices.CPUData()[i] = 
						IEMatrix4x4::LookAt(currentLight.position,
											currentLight.position + SceneLights::aLightDir[i],
											SceneLights::aLightUp[i]);
				}
				scene.getSceneLights().viewMatrices.SendData();

				// Put a 45 degree frustum projection matrix to the viewTransform part of the
				// FrameTransformUniform Buffer it'll be required on area light omni directional frustum
				viewTransform = IEMatrix4x4::Perspective(45.0f, 1.0f,
														 0.1f, currentLight.color.getW());
				projection = IEMatrix4x4::Perspective(90.0f, 1.0f,
													  0.1f, currentLight.color.getW());
				break;
			}
		}

		// Determine projection params
		// Do not waste objects that are out of the current view frustum
		cameraTransform.Update(FrameTransformBufferData
							   {
							   		viewTransform,
							   		projection
							   });

		// FBO Bind and render calls
		glBindFramebuffer(GL_FRAMEBUFFER, scene.getSceneLights().shadowMapFBOs[i]);
		glClear(GL_DEPTH_BUFFER_BIT);
		for(unsigned int i = 0; i < scene.DrawCount(); i++)
		{
			scene.getDrawBuffer().getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);
			glDrawElementsIndirect(GL_TRIANGLES,
								   GL_UNSIGNED_INT,
								   (void *) (i * sizeof(DrawPointIndexed)));
		}
	}
}

void DeferredRenderer::GPass(SceneI& scene,
							 const Camera& camera)
{
	gBuffer.BindAsFBO();
	gBuffer.AlignViewport();

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDepthMask(false);
	glDepthFunc(GL_EQUAL);
	glColorMask(true, true, true, true);
	glClear(GL_COLOR_BUFFER_BIT);

	// Camera Transform
	cameraTransform.Update(camera.generateTransform());
	cameraTransform.Bind();

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertexGBufferWrite.Bind();
	fragmentGBufferWrite.Bind();

	// DrawCall
	DrawBuffer& dBuffer = scene.getDrawBuffer();
	scene.getGPUBuffer().Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();

	for(unsigned int i = 0; i < scene.DrawCount(); i++)
	{
		dBuffer.BindMaterialForDraw(i);
		dBuffer.getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));
	}
}

void DeferredRenderer::LightPass(SceneI& scene, const Camera& camera)
{
	// Light pass
	// Texture Binds
	glActiveTexture(GL_TEXTURE0 + T_SHADOW);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, scene.getSceneLights().lightShadowMaps);
	gBuffer.BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	gBuffer.BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
	gBuffer.BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	glBindSampler(T_COLOR, flatSampler);
	glBindSampler(T_NORMAL, flatSampler);
	glBindSampler(T_DEPTH, flatSampler);
	glBindSampler(T_SHADOW, shadowMapSampler);

	// Buffer Binds
	FrameTransformBufferData ft = camera.generateTransform();
	cameraTransform.Update(ft);
	cameraTransform.Bind();
	scene.getSceneLights().lightsGPU.BindAsShaderStorageBuffer(LU_LIGHT);

	// Inverse Frame Transforms
	invFrameTransform.BindAsUniformBuffer(U_INVFTRANSFORM);
	float depthRange[2];
	glGetFloatv(GL_DEPTH_RANGE, depthRange);
	invFrameTransform.CPUData()[0] = InvFrameTransform
	{		
		ft.view.Inverse() * ft.projection.Inverse(),
		IEVector4(camera.pos),
		{0, 0, gBuffWidth, gBuffHeight},
		{depthRange[0], depthRange[1], 0.0f, 0.0f}
	};
	invFrameTransform.SendData();

	// Bind LightIntensity Buffer as framebuffer
	glBindFramebuffer(GL_FRAMEBUFFER, lightIntensityFBO);
	glViewport(0, 0, gBuffWidth, gBuffHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	// Get VAO from Scene Lights
	Shader::Unbind(ShaderType::GEOMETRY);
	vertLightPass.Bind();
	fragLightPass.Bind();

	// Open Additive Blending
	// Intensity of different lights will be added
	// Only render backfaces to eliminate multi fragment from light enclosure objects
	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_GREATER);
	glDepthMask(false);

	scene.getSceneLights().lightDrawParams.BindAsDrawIndirectBuffer();
	glBindVertexArray(scene.getSceneLights().lightVAO);
	glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, 3, sizeof(DrawPointIndexed));

	glFrontFace(GL_CCW);
	glDisable(GL_BLEND);
}

void DeferredRenderer::DPass(SceneI& scene, const Camera& camera)
{
	// Depth Pass that to utilize early fragment test
	gBuffer.BindAsFBO();
	gBuffer.AlignViewport();

	// States
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDepthMask(true);
	glColorMask(false, false, false, false);
	glDepthFunc(GL_LESS);
	glClear(GL_DEPTH_BUFFER_BIT |
			GL_STENCIL_BUFFER_BIT);
	
	// Shaders
	fragShadowMap.Bind();
	vertDPass.Bind();
	Shader::Unbind(ShaderType::GEOMETRY);

	// Camera Transform
	cameraTransform.Update(camera.generateTransform());
	cameraTransform.Bind();

	// Buffers
	DrawBuffer& dBuffer = scene.getDrawBuffer();
	scene.getGPUBuffer().Bind();
	dBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();

	for(unsigned int i = 0; i < scene.DrawCount(); i++)
	{
		dBuffer.getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));
	}
}

void DeferredRenderer::LightMerge(const Camera& camera)
{
	// Render to main framebuffer as post process
	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragLightApply.Bind();

	// Texture Binds
	gBuffer.BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	glActiveTexture(GL_TEXTURE0 + T_INTENSITY);
	glBindTexture(GL_TEXTURE_2D, lightIntensityTex);
	glBindSampler(T_COLOR, linearSampler);
	glBindSampler(T_INTENSITY, linearSampler);

	// FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, 
			   static_cast<GLsizei>(camera.width), 
			   static_cast<GLsizei>(camera.height));

	// States
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT);
	glDepthMask(false);
	glColorMask(true, true, true, true);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);

	// Bind Post Process VAO
	glBindVertexArray(postProcessTriVao);

	// DrawCall
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::Render(SceneI& scene, const Camera& camera)
{
	// Shadow Map Generation
	// Calculate Frustum Parameters from Render Camera
	float tanHalfFovX = IEMath::TanF(IEMath::ToRadians(camera.fovX * 0.5f));
	float aspectRatio = camera.width / camera.height;
	IEVector3 camDir = (camera.centerOfInterest - camera.pos).NormalizeSelf();
	IEVector3 right = camDir.CrossProduct(camera.up).NormalizeSelf();

	float farHalfWidth = camera.far * tanHalfFovX;
	float farHalfHeight = farHalfWidth / aspectRatio;

	// Plane Center Points
	IEVector3 planeCenterFar = camera.pos + camDir * camera.far;

	//IEVector3 farTopLeft = planeCenterFar + (mainRenderCamera.up * farHeight * 0.5f) - (right * farWidth * 0.5f);
	IEVector3 farTopRight = planeCenterFar + (camera.up * farHalfHeight) + (right * farHalfWidth);
	IEVector3 farBottomLeft = planeCenterFar - (camera.up * farHalfHeight) - (right * farHalfWidth);
	IEVector3 farBottomRight = planeCenterFar - (camera.up * farHalfHeight) + (right * farHalfWidth);

	// MRectangular Prism View Frustum Coords in World Space
	const IEVector3 span[3] =
	{
		farTopRight - farBottomRight,
		-camera.far * camDir,
		farBottomLeft - farBottomRight
	};
	RectPrism viewFrustum(span, farBottomRight);

	// Shadow Map Generation
	GenerateShadowMaps(scene, camera, viewFrustum);

	// Depth Pre-Pass
	DPass(scene, camera);

	// Actual Render
	// G Pass
	GPass(scene, camera);

	// Light Pass
	LightPass(scene, camera);
	
	// Light Intensity Merge
	LightMerge(camera);

	// All Done!
}