#include "SceneLights.h"
#include "IEUtility/IEVector3.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"
#include "Globals.h"
#include "FrameTransformBuffer.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEMath.h"
#include "GFG/GFGFileLoader.h"

const GLsizei SceneLights::shadowMapWH = /*512;*/1024;/*1536;*//*2048;*///4096;

const uint32_t SceneLights::numShadowCascades = 4;
const uint32_t SceneLights::shadowMipCount = 8;
const uint32_t SceneLights::mipSampleCount = 3;
const char* SceneLights::lightAOIFileName = "lightAOI.gfg";
GLuint SceneLights::lightShapeBuffer = 0;
GLuint SceneLights::lightShapeIndexBuffer = 0;
DrawPointIndexed SceneLights::drawParamsGeneric[3] = {{0}, {0}, {0}};

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
	-IEVector3::Yaxis,
	-IEVector3::Yaxis,
	IEVector3::Zaxis,
	-IEVector3::Zaxis,
	-IEVector3::Yaxis,
	-IEVector3::Yaxis
};

const IEVector3 SceneLights::aLightDir[6] =
{
	IEQuaternion(IEMath::ToRadians(22.5f), -IEVector3::Zaxis).ApplyRotation(IEVector3::Xaxis),
	IEQuaternion(IEMath::ToRadians(22.5f), IEVector3::Zaxis).ApplyRotation(-IEVector3::Xaxis),
	IEVector3::ZeroVector,	// Unused
	-IEVector3::Yaxis,
	IEQuaternion(IEMath::ToRadians(22.5f), IEVector3::Xaxis).ApplyRotation(IEVector3::Zaxis),
	IEQuaternion(IEMath::ToRadians(22.5f), -IEVector3::Xaxis).ApplyRotation(-IEVector3::Zaxis)
};

const IEVector3 SceneLights::aLightUp[6] =
{
	IEQuaternion(IEMath::ToRadians(22.5f), -IEVector3::Zaxis).ApplyRotation(-IEVector3::Yaxis),
	IEQuaternion(IEMath::ToRadians(22.5f), IEVector3::Zaxis).ApplyRotation(-IEVector3::Yaxis),
	IEVector3::ZeroVector,	// Unused
	-IEVector3::Zaxis,
	IEQuaternion(IEMath::ToRadians(22.5f), IEVector3::Xaxis).ApplyRotation(-IEVector3::Yaxis),
	IEQuaternion(IEMath::ToRadians(22.5f), -IEVector3::Xaxis).ApplyRotation(-IEVector3::Yaxis)
};

SceneLights::SceneLights(const Array32<Light>& lights)
	: lightsGPU(lights.length)
	, lightShadowMaps(0)
	, lightViewProjMatrices(6 * lights.length)
	, lightDrawParams(3)
	, lightIndexBuffer(lights.length)
	, lightProjMatrices(6 * lights.length)
	, lightInvViewProjMatrices(6 * lights.length)
{
	lightViewProjMatrices.RecieveData(6 * lights.length);
	glGenTextures(1, &lightShadowMaps);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, lightShadowMaps);
	glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, shadowMipCount, GL_R32F, shadowMapWH, shadowMapWH, 6 * lights.length);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_GREATER);
	
	shadowMapFBOs.resize(lights.length);
	shadowMapViews.resize(lights.length);
	glGenFramebuffers(lights.length, shadowMapFBOs.data());
	glGenTextures(lights.length, shadowMapViews.data());

	// Depth Tex Helper
	glGenTextures(1, &shadowMapCubeDepth);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowMapCubeDepth);
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_DEPTH_COMPONENT32F, shadowMapWH, shadowMapWH);

	// Interpret CubemapTexture Array as 2D Texture
	// Used for Directional Lights (each cube side is a cascade)
	glGenTextures(1, &shadowMapArrayView);
	glTextureView(shadowMapArrayView, GL_TEXTURE_2D_ARRAY, lightShadowMaps, GL_R32F,
				  0, shadowMipCount, 0, 6 * lights.length);

	for(unsigned int i = 0; i < lights.length; i++)
	{
		lightsGPU.AddData(lights.arr[i]);
		lightShadowCast.push_back(true);
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBOs[i]);
		glTextureView(shadowMapViews[i], GL_TEXTURE_CUBE_MAP, lightShadowMaps, GL_R32F,
					  0, 1, 6 * i, 6);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadowMapViews[i], 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowMapCubeDepth, 0);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	}		
	lightsGPU.SendData();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	if(lightShapeBuffer == 0)
	{
		std::ifstream stream(lightAOIFileName, std::ios_base::in | std::ios_base::binary);
		GFGFileReaderSTL stlFileReader(stream);
		GFGFileLoader gfgFile(&stlFileReader);
		std::vector<DrawPointIndexed> drawCalls;
		gfgFile.ValidateAndOpen();

		assert(gfgFile.Header().meshes.size() == 3);
		std::vector<uint8_t> vData(gfgFile.AllMeshVertexDataSize());
		std::vector<uint8_t> viData(gfgFile.AllMeshIndexDataSize());
		gfgFile.AllMeshVertexData(vData.data());
		gfgFile.AllMeshIndexData(viData.data());

		glGenBuffers(1, &lightShapeBuffer);
		glBindBuffer(GL_COPY_WRITE_BUFFER, lightShapeBuffer);
		glBufferData(GL_COPY_WRITE_BUFFER, vData.size(), vData.data(), GL_STATIC_DRAW);

		glGenBuffers(1, &lightShapeIndexBuffer);
		glBindBuffer(GL_COPY_WRITE_BUFFER, lightShapeIndexBuffer);
		glBufferData(GL_COPY_WRITE_BUFFER, viData.size(), viData.data(), GL_STATIC_DRAW);

		uint32_t vOffset = 0, viOffset = 0;
		uint32_t i = 0;
		for(const GFGMeshHeader& mesh : gfgFile.Header().meshes)
		{
			assert(mesh.headerCore.indexSize == sizeof(uint32_t));

			drawParamsGeneric[i].baseInstance = 0;
			drawParamsGeneric[i].baseVertex = vOffset;
			drawParamsGeneric[i].firstIndex = viOffset;
			drawParamsGeneric[i].count = static_cast<uint32_t>(mesh.headerCore.indexCount);
			drawParamsGeneric[i].instanceCount = 0;

			vOffset += static_cast<uint32_t>(mesh.headerCore.vertexCount);
			viOffset += static_cast<uint32_t>(mesh.headerCore.indexCount);
			i++;
		}
	}

	// Light Draw Param Generation
	uint32_t dCount = 0, aCount = 0, pCount = 0, i = 0;
	uint32_t dIndex = 0, aIndex = 0, pIndex = 0;
	std::vector<uint32_t>& lIndexBuff = lightIndexBuffer.CPUData();
	lIndexBuff.resize(lights.length);
	for(const Light& l : lightsGPU.CPUData())
	{
		if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::AREA)))
			aCount++;
		else if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::DIRECTIONAL)))
			dCount++;
		else if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::POINT)))
			pCount++;
	}
	for(const Light& l : lightsGPU.CPUData())
	{
		if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::AREA)))
		{
			lIndexBuff[pCount + dCount + aIndex] = i;
			aIndex++;
		}
		else if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::DIRECTIONAL)))
		{
			lIndexBuff[pCount + dIndex] = i;
			dIndex++;
		}
		else if(l.position.getW() == static_cast<float>(static_cast<int>(LightType::POINT)))
		{
			lIndexBuff[0 + pIndex] = i;
			pIndex++;
		}
		i++;
	}
	lightIndexBuffer.SendData();

	// Draw Buffers
	lightDrawParams.AddData(drawParamsGeneric[static_cast<int>(LightType::POINT)]);
	lightDrawParams.AddData(drawParamsGeneric[static_cast<int>(LightType::DIRECTIONAL)]);
	lightDrawParams.AddData(drawParamsGeneric[static_cast<int>(LightType::AREA)]);

	lightDrawParams.CPUData()[static_cast<int>(LightType::POINT)].instanceCount = pCount;
	lightDrawParams.CPUData()[static_cast<int>(LightType::DIRECTIONAL)].instanceCount = dCount;
	lightDrawParams.CPUData()[static_cast<int>(LightType::AREA)].instanceCount = aCount;
	lightDrawParams.CPUData()[static_cast<int>(LightType::POINT)].baseInstance = 0;
	lightDrawParams.CPUData()[static_cast<int>(LightType::DIRECTIONAL)].baseInstance = pCount;
	lightDrawParams.CPUData()[static_cast<int>(LightType::AREA)].baseInstance = pCount + dCount;
	lightDrawParams.SendData();

	// Create VAO
	// PostProcess VAO
	glGenVertexArrays(1, &lightVAO);
	glBindVertexArray(lightVAO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, lightShapeIndexBuffer);

	// Pos
	glBindVertexBuffer(0, lightShapeBuffer, 0, sizeof(float) * 3);
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS, 3, GL_FLOAT, false, 0);
	glVertexAttribBinding(IN_POS, 0);

	// Index
	glBindVertexBuffer(1, lightIndexBuffer.getGLBuffer(), 0, sizeof(uint32_t));
	glVertexBindingDivisor(1, 1);
	glEnableVertexAttribArray(IN_LIGHT_INDEX);
	glVertexAttribIFormat(IN_LIGHT_INDEX, 1, GL_UNSIGNED_INT, 0);
	glVertexAttribBinding(IN_LIGHT_INDEX, 1);

	assert(lightsGPU.CPUData().size() == lightShadowCast.size());
}

SceneLights::~SceneLights()
{
	glDeleteTextures(1, &shadowMapCubeDepth);
	glDeleteTextures(1, &lightShadowMaps);
	glDeleteFramebuffers(static_cast<GLsizei>(shadowMapFBOs.size()), shadowMapFBOs.data());
	glDeleteTextures(static_cast<GLsizei>(shadowMapViews.size()), shadowMapViews.data());
	glDeleteVertexArrays(1, &lightVAO);
	glDeleteTextures(1, &shadowMapArrayView);
}

uint32_t SceneLights::Count() const
{
	return static_cast<uint32_t>(lightShadowCast.size());
}

GLuint SceneLights::GetLightBufferGL()
{
	return lightsGPU.getGLBuffer();
}

GLuint SceneLights::GetShadowArrayGL()
{
	return shadowMapArrayView;
}

GLuint SceneLights::GetVPMatrixGL()
{
	return lightViewProjMatrices.getGLBuffer();
}

const std::vector<IEMatrix4x4>&	SceneLights::GetLightProjMatrices()
{
	return lightProjMatrices;
}

const std::vector<IEMatrix4x4>& SceneLights::GetLightInvViewProjMatrices()
{
	return lightInvViewProjMatrices;
}

void SceneLights::ChangeLightPos(uint32_t index, IEVector3 position)
{
	Light l = lightsGPU.GetData(index);
	l.position.setX(position.getX());
	l.position.setY(position.getY());
	l.position.setZ(position.getZ());
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
	l.direction.setW(radius);
	lightsGPU.ChangeData(index, l);
}

void SceneLights::ChangeLightIntensity(uint32_t index, float intensity)
{
	Light l = lightsGPU.GetData(index);
	l.color.setW(intensity);
	lightsGPU.ChangeData(index, l);
}

void SceneLights::ChangeLightShadow(uint32_t index, bool shadowStatus)
{
	lightShadowCast[index] = shadowStatus;
}

IEVector3 SceneLights::GetLightPos(uint32_t index) const
{
	return lightsGPU.CPUData()[index].position;
}

LightType SceneLights::GetLightType(uint32_t index) const
{
	return static_cast<LightType>(static_cast<int>(
			lightsGPU.CPUData()[index].position.getW()));
}

IEVector3 SceneLights::GetLightDir(uint32_t index) const
{
	return lightsGPU.CPUData()[index].direction;
}

IEVector3 SceneLights::GetLightColor(uint32_t index) const
{
	return lightsGPU.CPUData()[index].color;
}

float SceneLights::GetLightRadius(uint32_t index) const
{
	return lightsGPU.CPUData()[index].direction.getW();
}

float SceneLights::GetLightIntensity(uint32_t index) const
{
	return lightsGPU.CPUData()[index].color.getW();
}

bool SceneLights::GetLightShadow(uint32_t index) const
{
	return lightShadowCast[index];
}