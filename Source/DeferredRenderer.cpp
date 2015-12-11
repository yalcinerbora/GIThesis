#include "DeferredRenderer.h"
#include "IEUtility/IEMath.h"
#include "Scene.h"
#include "Globals.h"
#include "Camera.h"
#include "RectPrism.h"
#include "DrawBuffer.h"

const GLsizei DeferredRenderer::gBuffWidth = /*160;*//*320;*//*640;*//*800;*/1280;/*1920*/;//3840;
const GLsizei DeferredRenderer::gBuffHeight = /*90;*//*180;*//*360;*//*450;*/720;/*1080;*///2160;

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
	, fragPPGeneric(ShaderType::FRAGMENT, "Shaders/PProcessGeneric.frag")
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
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, gBuffer.getDepthGL(), 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);

	// SRGB Tex
	glGenTextures(1, &sRGBEndTex);
	glGenFramebuffers(1, &sRGBEndFBO);
	glBindTexture(GL_TEXTURE_2D, sRGBEndTex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_SRGB8_ALPHA8, gBuffWidth, gBuffHeight);

	glBindFramebuffer(GL_FRAMEBUFFER, sRGBEndFBO);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, sRGBEndTex, 0);
	assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	GLint encoding;
	glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, 
										  GL_COLOR_ATTACHMENT0,
										  GL_FRAMEBUFFER_ATTACHMENT_COLOR_ENCODING, 
										  &encoding);
	assert(encoding == GL_SRGB);
	
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
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_COMPARE_FUNC, GL_LESS);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glSamplerParameteri(shadowMapSampler, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);

	GLfloat col[] = { 1.0f, 0.0f, 0.0f, 0.0f };
	glSamplerParameterfv(shadowMapSampler, GL_TEXTURE_BORDER_COLOR, col);
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

GLuint DeferredRenderer::GetLightIntensityBufferGL()
{
	return lightIntensityTex;
}

InvFrameTransformBuffer& DeferredRenderer::GetInvFTransfrom()
{
	return invFrameTransform;
}

FrameTransformBuffer& DeferredRenderer::GetFTransform()
{
	return cameraTransform;
}

float DeferredRenderer::CalculateCascadeLength(float frustumFar,
											   unsigned int cascadeNo)
{
	// Geometric sum
	static const float exponent = 1.2f;
	float chunkSize = (std::powf(exponent, static_cast<float>(SceneLights::numShadowCascades)) - 1.0f) / (exponent - 1.0f);
	return std::powf(exponent, static_cast<float>(cascadeNo)) * (frustumFar / chunkSize);
}

BoundingSphere DeferredRenderer::CalculateShadowCascasde(float cascadeNear,
														 float cascadeFar,
														 const Camera& camera,
														 const IEVector3& lightDir)
{
	float cascadeDiff = cascadeFar - cascadeNear;

	// Shadow Map Generation
	// Calculate Frustum Parameters from Render Camera
	float tanHalfFovX = IEMath::TanF(IEMath::ToRadians(camera.fovX * 0.5f));
	float aspectRatio = camera.width / camera.height;
	IEVector3 camDir = (camera.centerOfInterest - camera.pos).NormalizeSelf();
	IEVector3 right = camDir.CrossProduct(camera.up).NormalizeSelf();
	IEVector3 camUp = camDir.CrossProduct(right).NormalizeSelf();

	float farHalfWidth = cascadeFar * tanHalfFovX;
	float farHalfHeight = farHalfWidth / aspectRatio;

	// Plane Center Points
	IEVector3 planeCenterFar = camera.pos + camDir * cascadeFar;

	IEVector3 farTopRight = planeCenterFar + (camUp * farHalfHeight) + (right * farHalfWidth);
	IEVector3 farBottomLeft = planeCenterFar - (camUp * farHalfHeight) - (right * farHalfWidth);
	IEVector3 farBottomRight = planeCenterFar - (camUp * farHalfHeight) + (right * farHalfWidth);

	// Frustum Span (sized)
	const IEVector3 span[3] =
	{
		farTopRight - farBottomRight,
		-cascadeDiff * camDir,
		farBottomLeft - farBottomRight
	};

	// Converting to bounding sphere
	float diam = (span[0] + span[1] + span[2]).Length();
	float radius = diam * 0.5f;
	IEVector3 centerPoint = farBottomRight + radius * (span[0] + span[1] + span[2]).NormalizeSelf();
	return BoundingSphere{centerPoint, radius};
}

void DeferredRenderer::GenerateShadowMaps(SceneI& scene,
										  const Camera& camera)
{
	fragShadowMap.Bind();
	vertShadowMap.Bind();

	// State
	// Rendering with polygon offset to eliminate shadow acne
	glColorMask(false, false, false, false);
	glDepthMask(true);
	glDepthFunc(GL_LESS);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glViewport(0, 0, SceneLights::shadowMapWH, SceneLights::shadowMapWH);

	// Render From Dir of the light	with proper view params
	for(int i = 0; i < scene.getSceneLights().lightsGPU.CPUData().size(); i++)
	{
		const Light& currentLight = scene.getSceneLights().lightsGPU.CPUData()[i];
		// Determine light type
		LightType t = static_cast<LightType>(static_cast<uint32_t>(currentLight.position.getW()));
		switch(t)
		{
			case LightType::POINT:
			{
				// Each Side will have 90 degree FOV
				// Geom shader will render for each layer
				IEMatrix4x4 projection = IEMatrix4x4::Perspective(90.0f, 1.0f,
																  0.1f, currentLight.color.getW());
				for(unsigned int j = 0; j < 6; j++)
				{
					IEMatrix4x4 view = IEMatrix4x4::LookAt(currentLight.position,
														   currentLight.position + SceneLights::pLightDir[j],
														   SceneLights::pLightUp[j]);
					scene.getSceneLights().lightViewProjMatrices.CPUData()[i * 6 + j] = projection * view;
				}
				break;
			}
			case LightType::DIRECTIONAL:
			{
				for(unsigned int j = 0; j < SceneLights::numShadowCascades; j++)
				{
					float cascade = CalculateCascadeLength(camera.far, j);
					BoundingSphere viewSphere = CalculateShadowCascasde(cascade * j,
																		cascade * (j + 1),
																		camera,
																		currentLight.direction);

					// Squre Orto Projection
					float radius = viewSphere.radius;
					IEMatrix4x4 projection = IEMatrix4x4::Ortogonal(//360.0f, -360.0f,
																	//-230.0f, 230.0f,
																	-radius, radius,
																	radius, -radius,
																	-1000.0f, 1000.0f);

					IEMatrix4x4 view = IEMatrix4x4::LookAt(viewSphere.center * IEVector3(1.0f, 1.0f, 1.0f),
														   viewSphere.center * IEVector3(1.0f, 1.0f, 1.0f) + currentLight.direction,
														   camera.up);

					// To eliminate shadow shimmering only change pixel sized frusutm changes
					IEVector3 unitPerTexel = (2.0f * IEVector3(radius, radius, radius)) / IEVector3(static_cast<float>(SceneLights::shadowMapWH), static_cast<float>(SceneLights::shadowMapWH), static_cast<float>(SceneLights::shadowMapWH));
					IEVector3 translatedOrigin = view * IEVector3::ZeroVector;
					IEVector3 texelTranslate;
					texelTranslate.setX(fmod(translatedOrigin.getX(), unitPerTexel.getX()));
					texelTranslate.setY(fmod(translatedOrigin.getY(), unitPerTexel.getY()));
					texelTranslate.setZ(fmod(translatedOrigin.getZ(), unitPerTexel.getZ()));
					texelTranslate = unitPerTexel - texelTranslate;
					//texelTranslate.setZ(0.0f);

					IEMatrix4x4 texelTranslateMatrix = IEMatrix4x4::Translate(texelTranslate);

					scene.getSceneLights().lightViewProjMatrices.CPUData()[i * 6 + j] = projection * texelTranslateMatrix * view;
				}
				break;
			}
			case LightType::AREA:
			{
				IEMatrix4x4 projections[2] = { IEMatrix4x4::Perspective(45.0f, 1.0f,
																		0.1f, currentLight.color.getW()),
											   IEMatrix4x4::Perspective(90.0f, 1.0f,
																		0.1f, currentLight.color.getW())};

				// we'll use 5 sides but each will comply different ares that a point light
				for(unsigned int j = 0; j < 6; j++)
				{
					uint32_t projIndex = (j == 3) ? 1 : 0;
					IEMatrix4x4 view = IEMatrix4x4::LookAt(currentLight.position,
														   currentLight.position + SceneLights::aLightDir[j],
														   SceneLights::aLightUp[j]);

					scene.getSceneLights().lightViewProjMatrices.CPUData()[i * 6 + j] = projections[projIndex] * view;	
				}	
				break;
			}
		}	
	}
	scene.getSceneLights().lightViewProjMatrices.SendData();

	// Binding
	cameraTransform.Bind();
	scene.getSceneLights().lightViewProjMatrices.BindAsShaderStorageBuffer(LU_LIGHT_MATRIX);

	// Render Loop
	for(int i = 0; i < scene.getSceneLights().lightsGPU.CPUData().size(); i++)
	{
		// FBO Bind and render calls
		glBindFramebuffer(GL_FRAMEBUFFER, scene.getSceneLights().shadowMapFBOs[i]);
		glClear(GL_DEPTH_BUFFER_BIT);

		if(!scene.getSceneLights().lightShadowCast[i])
			continue;

		// Draw Batches
		Array32<MeshBatchI*> batches = scene.getBatches();
		for(unsigned int j = 0; j < batches.length; j++)
		{
			GPUBuffer& currentGPUBuffer = batches.arr[j]->getGPUBuffer();
			DrawBuffer& currentDrawBuffer = batches.arr[j]->getDrawBuffer();

			currentGPUBuffer.Bind();
			currentDrawBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();

			// Base poly offset gives ok results on point-area lights
			glPolygonOffset(2.64f, 512.0f);

			const Light& currentLight = scene.getSceneLights().lightsGPU.CPUData()[i];
			LightType t = static_cast<LightType>(static_cast<uint32_t>(currentLight.position.getW()));
			switch(t)
			{
				case LightType::POINT: geomPointShadowMap.Bind(); break;
				case LightType::DIRECTIONAL: geomDirShadowMap.Bind();
					glPolygonOffset(4.12f, 1024.0f); // Higher offset req since camera span is large
					break;
				case LightType::AREA: geomAreaShadowMap.Bind(); break;
			}
			glUniform1ui(U_LIGHT_ID, static_cast<GLuint>(i));

			currentDrawBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);

			glMultiDrawElementsIndirect(GL_TRIANGLES,
										GL_UNSIGNED_INT,
										nullptr,
										static_cast<GLsizei>(batches.arr[j]->DrawCount()),
										sizeof(DrawPointIndexed));

			// Stays Here for Debugging purposes (nsight states)
			//for(unsigned int k = 0; k < batches.arr[j]->DrawCount(); k++)
			//{
			//	glDrawElementsIndirect(GL_TRIANGLES,
			//						   GL_UNSIGNED_INT,
			//						   (void *) (k * sizeof(DrawPointIndexed)));
			//}
		}
	}
	glDisable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(0.0f, 0.0f);
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
	// Draw Batches
	Array32<MeshBatchI*> batches = scene.getBatches();
	for(unsigned int i = 0; i < batches.length; i++)
	{
		GPUBuffer& currentGPUBuffer = batches.arr[i]->getGPUBuffer();
		DrawBuffer& currentDrawBuffer = batches.arr[i]->getDrawBuffer();

		currentGPUBuffer.Bind();
		currentDrawBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
		currentDrawBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);

		for(unsigned int j = 0; j < batches.arr[i]->DrawCount(); j++)
		{
			currentDrawBuffer.BindMaterialForDraw(j);
			glDrawElementsIndirect(GL_TRIANGLES,
								   GL_UNSIGNED_INT,
								   (void *)(j * sizeof(DrawPointIndexed)));
		}
	}
}

void DeferredRenderer::LightPass(SceneI& scene, const Camera& camera)
{
	// Light pass
	// Texture Binds
	glActiveTexture(GL_TEXTURE0 + T_SHADOW);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, scene.getSceneLights().lightShadowMaps);
	glActiveTexture(GL_TEXTURE0 + T_SHADOW_DIR);
	glBindTexture(GL_TEXTURE_2D_ARRAY, scene.getSceneLights().shadowMapArrayView);
	gBuffer.BindAsTexture(T_COLOR, RenderTargetLocation::COLOR);
	gBuffer.BindAsTexture(T_NORMAL, RenderTargetLocation::NORMAL);
	gBuffer.BindAsTexture(T_DEPTH, RenderTargetLocation::DEPTH);
	glBindSampler(T_COLOR, flatSampler);
	glBindSampler(T_NORMAL, flatSampler);
	glBindSampler(T_DEPTH, flatSampler);
	glBindSampler(T_SHADOW, shadowMapSampler);
	glBindSampler(T_SHADOW_DIR, shadowMapSampler);

	// Buffer Binds
	FrameTransformBufferData ft = camera.generateTransform();
	cameraTransform.Update(ft);
	cameraTransform.Bind();
	scene.getSceneLights().lightsGPU.BindAsShaderStorageBuffer(LU_LIGHT);

	// Inverse Frame Transforms
	invFrameTransform.BindAsUniformBuffer(U_INVFTRANSFORM);
	RefreshInvFTransform(camera, gBuffWidth, gBuffHeight);

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
	glEnable(GL_DEPTH_CLAMP);
	glFrontFace(GL_CW);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glDepthFunc(GL_GREATER);
	glDepthMask(false);

	scene.getSceneLights().lightDrawParams.BindAsDrawIndirectBuffer();
	glBindVertexArray(scene.getSceneLights().lightVAO);
	glMultiDrawElementsIndirect(GL_TRIANGLES, GL_UNSIGNED_INT, nullptr, 3, sizeof(DrawPointIndexed));

	glFrontFace(GL_CCW);
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
	glDisable(GL_DEPTH_CLAMP);
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

	
	
	// Draw Batches
	Array32<MeshBatchI*> batches = scene.getBatches();
	for(unsigned int i = 0; i < batches.length; i++)
	{
		GPUBuffer& currentGPUBuffer = batches.arr[i]->getGPUBuffer();
		DrawBuffer& currentDrawBuffer = batches.arr[i]->getDrawBuffer();

		currentGPUBuffer.Bind();
		currentDrawBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
		currentDrawBuffer.getModelTransformBuffer().BindAsShaderStorageBuffer(LU_MTRANSFORM);

		glMultiDrawElementsIndirect(GL_TRIANGLES,
									GL_UNSIGNED_INT,
									nullptr,
									static_cast<GLsizei>(batches.arr[i]->DrawCount()),
									sizeof(DrawPointIndexed));

		// Stays Here for Debugging purposes (nsight states)
		//for(unsigned int j = 0; j < batches.arr[i]->DrawCount(); j++)
		//{
		//	glDrawElementsIndirect(GL_TRIANGLES,
		//						   GL_UNSIGNED_INT,
		//						   (void *) (j * sizeof(DrawPointIndexed)));
		//}
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
	glBindSampler(T_COLOR, flatSampler);
	glBindSampler(T_INTENSITY, flatSampler);

	// FBO
	glBindFramebuffer(GL_FRAMEBUFFER, sRGBEndFBO);
	glViewport(0, 0, 
			   static_cast<GLsizei>(gBuffWidth),
			   static_cast<GLsizei>(gBuffHeight));

	// States
	glEnable(GL_FRAMEBUFFER_SRGB);
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
	
	// Passthrough to Default FBO
	//
	// SRGB Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D, sRGBEndTex);
	glBindSampler(T_COLOR, linearSampler);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragPPGeneric.Bind();

	// Default FBO
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	
	// States
	glDisable(GL_FRAMEBUFFER_SRGB);
	glClear(GL_COLOR_BUFFER_BIT);

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::RefreshInvFTransform(const Camera& camera,
											GLsizei width,
											GLsizei height)
{
	FrameTransformBufferData ft = camera.generateTransform();

	float depthRange[2];
	glGetFloatv(GL_DEPTH_RANGE, depthRange);
	invFrameTransform.CPUData()[0] = InvFrameTransform
	{
		ft.view.Inverse() * ft.projection.Inverse(),
		IEVector4(camera.pos.getX(), camera.pos.getY(), camera.pos.getZ(), CalculateCascadeLength(camera.far, 0)),
		IEVector4((camera.centerOfInterest - camera.pos).NormalizeSelf()),
		{0, 0, width, height},
		{ depthRange[0], depthRange[1], 0.0f, 0.0f }
	};
	invFrameTransform.SendData();
}

void DeferredRenderer::PopulateGBuffer(SceneI& scene, const Camera& camera)
{
	// Depth Pre-Pass
	DPass(scene, camera);

	// Actual Render
	// G Pass
	GPass(scene, camera);
}

void DeferredRenderer::Render(SceneI& scene, const Camera& camera)
{
	// Shadow Map Generation
	GenerateShadowMaps(scene, camera);

	// GPass
	PopulateGBuffer(scene, camera);
	
	// Light Pass
	LightPass(scene, camera);
	
	// Light Intensity Merge
	LightMerge(camera);

	// All Done!
}

void DeferredRenderer::ShowTexture(const Camera& camera, GLuint tex)
{
	// Only Draw Color Buffer
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0,
			   static_cast<GLsizei>(camera.width),
			   static_cast<GLsizei>(camera.height));
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0);

	glDisable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);
	glDepthMask(true);
	glColorMask(true, true, true, true);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Shaders
	Shader::Unbind(ShaderType::GEOMETRY);
	vertPPGeneric.Bind();
	fragPPGeneric.Bind();

	// Texture
	glActiveTexture(GL_TEXTURE0 + T_COLOR);
	glBindTexture(GL_TEXTURE_2D, tex);
	glBindSampler(T_COLOR, linearSampler);

	// VAO
	glBindVertexArray(postProcessTriVao);

	// Draw
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void DeferredRenderer::ShowColorGBuffer(const Camera& camera)
{
	ShowTexture(camera, gBuffer.getColorGL());
}

void DeferredRenderer::ShowLIBuffer(const Camera& camera)
{
	ShowTexture(camera, lightIntensityTex);
}
