#include "SceneLights.h"
#include "IEUtility/IEVector3.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"
#include "Globals.h"
#include "FrameTransformBuffer.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEMath.h"

const GLsizei SceneLights::shadowMapW = 512;
const GLsizei SceneLights::shadowMapH = 512;

const IEVector3 SceneLights::pLightDir[6] =
{
	IEVector3::Xaxis,
	-IEVector3::Xaxis,
	IEVector3::Yaxis,
	-IEVector3::Yaxis,
	IEVector3::Zaxis,
	-IEVector3::Zaxis
};

const IEVector3 SceneLights::pLightUp[6] =
{
	IEVector3::Yaxis,
	IEVector3::Yaxis,
	-IEVector3::Zaxis,
	IEVector3::Zaxis,
	IEVector3::Yaxis,
	IEVector3::Yaxis
};

const IEVector3 SceneLights::aLightDir[6] =
{
	IEQuaternion(IEMath::ToRadians(45.0f), -IEVector3::Zaxis).ApplyRotation(IEVector3::Xaxis),
	IEQuaternion(IEMath::ToRadians(45.0f), IEVector3::Zaxis).ApplyRotation(-IEVector3::Xaxis),
	IEVector3::ZeroVector,	// Unused
	-IEVector3::Yaxis,
	IEQuaternion(IEMath::ToRadians(45.0f), IEVector3::Xaxis).ApplyRotation(IEVector3::Zaxis),
	IEQuaternion(IEMath::ToRadians(45.0f), -IEVector3::Xaxis).ApplyRotation(-IEVector3::Zaxis)
};

const IEVector3 SceneLights::aLightUp[6] =
{
	IEQuaternion(IEMath::ToRadians(45.0f), -IEVector3::Zaxis).ApplyRotation(IEVector3::Yaxis),
	IEQuaternion(IEMath::ToRadians(45.0f), IEVector3::Zaxis).ApplyRotation(IEVector3::Yaxis),
	IEVector3::ZeroVector,	// Unused
	IEVector3::Zaxis,
	IEQuaternion(IEMath::ToRadians(45.0f), IEVector3::Xaxis).ApplyRotation(IEVector3::Yaxis),
	IEQuaternion(IEMath::ToRadians(45.0f), -IEVector3::Xaxis).ApplyRotation(IEVector3::Yaxis)
};

SceneLights::SceneLights(const Array32<Light>& lights)
	: lightsGPU(lights.length)
	, lightShadowMaps(0)
	, viewMatrices(6)
	, fragShadowMap(ShaderType::FRAGMENT, "Shaders/ShadowMap.frag")
	, vertShadowMap(ShaderType::VERTEX, "Shaders/ShadowMap.vert")
	, geomAreaShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapA.geom")
	, geomDirShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapD.geom")
	, geomPointShadowMap(ShaderType::GEOMETRY, "Shaders/ShadowMapP.geom")
{
	viewMatrices.RecieveData(6);
	glGenTextures(1, &lightShadowMaps);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, lightShadowMaps);
	glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 1, GL_DEPTH_COMPONENT24, shadowMapW, shadowMapH, 6 * lights.length);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_GREATER);
	
	shadowMapFBOs.resize(lights.length);
	shadowMapViews.resize(lights.length);
	glGenFramebuffers(lights.length, shadowMapFBOs.data());
	glGenTextures(lights.length, shadowMapViews.data());

	for(unsigned int i = 0; i < lights.length; i++)
	{
		lightsGPU.AddData(lights.arr[i]);
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBOs[i]);
		glTextureView(shadowMapViews[i], GL_TEXTURE_CUBE_MAP, lightShadowMaps, GL_DEPTH_COMPONENT24,
					  0, 1, 6 * i, 6);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowMapViews[i], 0);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	}		
	lightsGPU.SendData();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

SceneLights::~SceneLights()
{
	glDeleteTextures(1, &lightShadowMaps);
	glDeleteFramebuffers(static_cast<GLsizei>(shadowMapFBOs.size()), shadowMapFBOs.data());
	glDeleteTextures(static_cast<GLsizei>(shadowMapViews.size()), shadowMapViews.data());
}

void SceneLights::GenerateShadowMaps(DrawBuffer& drawBuffer, GPUBuffer& gpuBuffer,
									 FrameTransformBuffer& fTransform,
									 unsigned int drawCount,
									 IEVector3 wFrustumMin,
	 								 IEVector3 wFrustumMax)
{
	fragShadowMap.Bind();
	vertShadowMap.Bind();

	// State
	glColorMask(false, false, false, false);
	glDepthMask(true);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_CULL_FACE);
	glViewport(0, 0, shadowMapW, shadowMapH);

	gpuBuffer.Bind();
	fTransform.Bind();
	drawBuffer.getDrawParamBuffer().BindAsDrawIndirectBuffer();
	viewMatrices.BindAsUniformBuffer(U_SHADOW_VIEW);

	// Render From Dir of the light	with proper view params
	IEMatrix4x4 viewTransform = IEMatrix4x4::IdentityMatrix;
	IEMatrix4x4 projection;
	for(int i = 0; i < lightsGPU.CPUData().size(); i++)
	{
		const Light& currentLight = lightsGPU.CPUData()[i];
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
					viewMatrices.CPUData()[i] = IEMatrix4x4::LookAt(currentLight.position,
																	currentLight.position + pLightDir[i],
																	pLightUp[i]);
				}
				viewMatrices.SendData();
				projection = IEMatrix4x4::Perspective(90.0f, 1.0f,
													  0.1f, currentLight.color.getW() + 1000.0f);
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
				IEVector3 vFrustumMin = viewTransform * wFrustumMin;
				IEVector3 vFrustumMax = viewTransform * wFrustumMax;
				projection = IEMatrix4x4::Ortogonal(vFrustumMin.getX(), vFrustumMax.getX(),
													vFrustumMax.getY(), vFrustumMin.getY(),
													-500, 500);
				break;
			}
			case LightType::AREA:
			{
				// Render to cube but only 5 sides (6th side is not illuminated)
				geomAreaShadowMap.Bind();
				// we'll use 5 sides but each will comply different ares that a point light
				for(unsigned int i = 0; i < 6; i++)
				{
					viewMatrices.CPUData()[i] = IEMatrix4x4::LookAt(currentLight.position,
																	currentLight.position + aLightDir[i],
																	aLightUp[i]);
				}
				viewMatrices.SendData();

				// Put a 45 degree frustum projection matrix to the viewTransform part of the
				// FrameTransformUniform Buffer it'll be required on area light omni directional frustum
				viewTransform = IEMatrix4x4::Perspective(45.0f, 1.0f,
														 0.1f, currentLight.color.getW() + 10000.0f);
				projection = IEMatrix4x4::Perspective(90.0f, 1.0f, 
													  0.1f, currentLight.color.getW() + 10000.0f);
				break;
			}
		}

		// Determine projection params
		// Do not waste objects that are out of the current view frustum
		fTransform.Update
		(
			FrameTransformBufferData
			{
				viewTransform,
				projection,
				IEMatrix4x4::IdentityMatrix
			}
		);
		
		// FBO Bind and render calls
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBOs[i]);
		glClear(GL_DEPTH_BUFFER_BIT);
		for(unsigned int i = 0; i < drawCount; i++)
		{
			drawBuffer.getModelTransformBuffer().BindAsUniformBuffer(U_MTRANSFORM, i, 1);
			glDrawElementsIndirect(GL_TRIANGLES,
								   GL_UNSIGNED_INT,
								   (void *) (i * sizeof(DrawPointIndexed)));
		}
	}
}

void SceneLights::ChangeLightPos(uint32_t index, IEVector3 position)
{
	Light l = lightsGPU.GetData(index);
	l.position.setX(position.getX());
	l.position.setY(position.getY());
	l.position.setZ(position.getZ());
	lightsGPU.ChangeData(index, l);
}

void SceneLights::ChangeLightType(uint32_t index, LightType type)
{
	Light l = lightsGPU.GetData(index);
	l.position.setW(static_cast<float>(type));
	lightsGPU.ChangeData(index, l);
}

void SceneLights::ChangeLightDir(uint32_t index, IEVector3 direction)
{
	Light l = lightsGPU.GetData(index);
	l.direction.setX(direction.getX());
	l.direction.setY(direction.getY());
	l.direction.setZ(direction.getZ());
	lightsGPU.ChangeData(index, l);
}

void SceneLights::ChangeLightColor(uint32_t index, IEVector3 color)
{
	Light l = lightsGPU.GetData(index);
	l.color.setX(color.getX());
	l.color.setY(color.getY());
	l.color.setZ(color.getZ());
	lightsGPU.ChangeData(index, l);
}

void SceneLights::ChangeLightRadius(uint32_t index, float radius)
{
	Light l = lightsGPU.GetData(index);
	l.color.setW(radius);
	lightsGPU.ChangeData(index, l);
}
