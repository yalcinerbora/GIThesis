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
{
	viewMatrices.RecieveData(6);
	glGenTextures(1, &lightShadowMaps);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, lightShadowMaps);
	glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 1, GL_DEPTH_COMPONENT32, shadowMapW, shadowMapH, 6 * lights.length);
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
