#include "SceneLights.h"
#include "IEUtility/IEVector3.h"
#include "VertexBuffer.h"
#include "DrawBuffer.h"
#include "Globals.h"
#include "FrameTransformBuffer.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEMath.h"
#include "GFG/GFGFileLoader.h"
#include "Camera.h"
#include "DeferredRenderer.h"

const IEVector3 SceneLights::pLightDir[CubeSide] =
{
	IEVector3::XAxis,
	-IEVector3::XAxis,
	IEVector3::YAxis,
	-IEVector3::YAxis,
	IEVector3::ZAxis,
	-IEVector3::ZAxis
};

const IEVector3 SceneLights::pLightUp[CubeSide] =
{
	-IEVector3::YAxis,
	-IEVector3::YAxis,
	IEVector3::ZAxis,
	-IEVector3::ZAxis,
	-IEVector3::YAxis,
	-IEVector3::YAxis
};

const IEVector3 SceneLights::aLightDir[CubeSide] =
{
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), -IEVector3::ZAxis).ApplyRotation(IEVector3::XAxis),
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), IEVector3::ZAxis).ApplyRotation(-IEVector3::XAxis),
	IEVector3::ZeroVector,	// Unused
	-IEVector3::YAxis,
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), IEVector3::XAxis).ApplyRotation(IEVector3::ZAxis),
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), -IEVector3::XAxis).ApplyRotation(-IEVector3::ZAxis)
};

const IEVector3 SceneLights::aLightUp[CubeSide] =
{
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), -IEVector3::ZAxis).ApplyRotation(-IEVector3::YAxis),
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), IEVector3::ZAxis).ApplyRotation(-IEVector3::YAxis),
	IEVector3::ZeroVector,	// Unused
	-IEVector3::ZAxis,
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), IEVector3::XAxis).ApplyRotation(-IEVector3::YAxis),
	IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef * 22.5), -IEVector3::XAxis).ApplyRotation(-IEVector3::YAxis)
};

float SceneLights::CalculateCascadeLength(float frustumFar,
										  unsigned int cascadeNo)
{
	// Geometric sum
	static const float exponent = 1.2f;
	float chunkSize = (std::powf(exponent, static_cast<float>(LightDrawBuffer::DirectionalCascadesCount)) - 1.0f) / (exponent - 1.0f);
	return std::powf(exponent, static_cast<float>(cascadeNo)) * (frustumFar / chunkSize);
}

IEBoundingSphere SceneLights::CalculateShadowCascasde(float cascadeNear,
													  float cascadeFar,
													  const Camera& camera,
													  const IEVector3& lightDir)
{
	float cascadeDiff = cascadeFar - cascadeNear;

	// Shadow Map Generation
	// Calculate Frustum Parameters from Render Camera
	float tanHalfFovX = std::tan(static_cast<float>(IEMathConstants::DegToRadCoef * camera.fovX * 0.5f));
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
	return IEBoundingSphere(centerPoint, radius);
}

void SceneLights::GenerateMatrices(const Camera& camera)
{
	// Render From Dir of the light	with proper view params
	for(int i = 0; i < static_cast<int>(lights.size()); i++)
	{
		const Light& currentLight = lights[i];
		LightType t = static_cast<LightType>(static_cast<uint32_t>(currentLight.position.getW()));
		switch(t)
		{
			case LightType::POINT:
			{
				// Each Side will have 90 degree FOV
				// Geom shader will render for each layer
				IEMatrix4x4 projection = IEMatrix4x4::Perspective(90.0f, 1.0f,
																  0.1f, currentLight.color.getW());
				for(int j = 0; j < CubeSide; j++)
				{
					IEMatrix4x4 view = IEMatrix4x4::LookAt(currentLight.position,
														   currentLight.position + SceneLights::pLightDir[j],
														   SceneLights::pLightUp[j]);
					lightViewProjMatrices[i * 6 + j] = projection * view;
					lightProjMatrices[i * 6 + j] = projection;
					lightInvViewProjMatrices[i * 6 + j] = (projection * view).Inverse();
				}
				break;
			}
			case LightType::DIRECTIONAL:
			{
				for(int j = 0; j < LightDrawBuffer::DirectionalCascadesCount; j++)
				{
					float cascade = CalculateCascadeLength(camera.far, j);
					IEBoundingSphere viewSphere = CalculateShadowCascasde(cascade * j,
																		  cascade * (j + 1),
																		  camera,
																		  currentLight.direction);

					// Squre Orto Projection
					float radius = viewSphere.radius;
					IEMatrix4x4 projection = IEMatrix4x4::Ortogonal(//360.0f, -360.0f,
																	//-230.0f, 230.0f,
																	-radius, radius,
																	radius, -radius,
																	-800.0f, 800.0f);

					IEMatrix4x4 view = IEMatrix4x4::LookAt(viewSphere.center * IEVector3(1.0f, 1.0f, 1.0f),
														   viewSphere.center * IEVector3(1.0f, 1.0f, 1.0f) + currentLight.direction,
														   camera.up);

					// To eliminate shadow shimmering only change pixel sized frusutm changes
					IEVector3 unitPerTexel = (2.0f * IEVector3(radius, radius, radius)) / IEVector3(static_cast<float>(SceneLights::shadowMapWH), static_cast<float>(SceneLights::shadowMapWH), static_cast<float>(SceneLights::shadowMapWH));
					unitPerTexel *= static_cast<float>(1 << (LightDrawBuffer::ShadowMipSampleCount));
					IEVector3 translatedOrigin = view * IEVector3::ZeroVector;
					IEVector3 texelTranslate;
					texelTranslate.setX(std::fmod(translatedOrigin.getX(), unitPerTexel.getX()));
					texelTranslate.setY(std::fmod(translatedOrigin.getY(), unitPerTexel.getY()));
					texelTranslate.setZ(std::fmod(translatedOrigin.getZ(), unitPerTexel.getZ()));
					texelTranslate = unitPerTexel - texelTranslate;
					//texelTranslate.setZ(0.0f);

					IEMatrix4x4 texelTranslateMatrix = IEMatrix4x4::Translate(texelTranslate);
					lightViewProjMatrices[i * 6 + j] = projection * texelTranslateMatrix * view;
					lightProjMatrices[i * 6 + j] = projection;
					lightInvViewProjMatrices[i * 6 + j] = (projection * texelTranslateMatrix * view).Inverse();
				}
				break;
			}
			case LightType::RECTANGULAR:
			{
				IEMatrix4x4 projections[2] = 
				{
					IEMatrix4x4::Perspective(45.0f, 1.0f, 0.1f, currentLight.color.getW()),
					IEMatrix4x4::Perspective(90.0f, 1.0f, 0.1f, currentLight.color.getW())
				};

				// we'll use 5 sides but each will comply different ares that a point light
				for(unsigned int j = 0; j < 6; j++)
				{
					uint32_t projIndex = (j == 3) ? 1 : 0;
					IEMatrix4x4 view = IEMatrix4x4::LookAt(currentLight.position,
														   currentLight.position + SceneLights::aLightDir[j],
														   SceneLights::aLightUp[j]);

					lightViewProjMatrices[i * 6 + j] = projections[projIndex] * view;
					lightProjMatrices[i * 6 + j] = projections[projIndex];
					lightInvViewProjMatrices[i * 6 + j] = (projections[projIndex] * view).Inverse();
				}
				break;
			}
		}
	}
}

SceneLights::SceneLights()
	: lightOffset(0)
	, matrixOffset(0)
	, lightShadowMaps(0)
	, shadowMapArrayView(0)
	, shadowMapCubeDepth(0)
{}

SceneLights::SceneLights(const std::vector<Light>& lights)
	: gpuBuffer(lights.size() * (sizeof(Light) +  sizeof(IEMatrix4x4) * CubeSide))
	, lightOffset(0)
	, matrixOffset(0)
	, lightShadowMaps(0)
	, shadowMapArrayView(0)
	, shadowMapCubeDepth(0)
	, shadowMapViews(lights.size())
	, shadowMapFBOs(lights.size())
	, lights(lights)
	, lightViewProjMatrices(lights.size() * CubeSide)
	, lightProjMatrices(lights.size() * CubeSide)
	, lightInvViewProjMatrices(lights.size() * CubeSide)
	, lightShadowCast(lights.size(), true)
{
	GLsizei lightCount = static_cast<GLsizei>(lights.size());

	glGenTextures(1, &lightShadowMaps);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, lightShadowMaps);
	glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, LightDrawBuffer::ShadowMapMipCount, 
				   GL_R32F, LightDrawBuffer::ShadowMapWH, LightDrawBuffer::ShadowMapWH,
				   CubeSide * lightCount);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_GREATER);
	
	glGenFramebuffers(lightCount, shadowMapFBOs.data());
	glGenTextures(lightCount, shadowMapViews.data());

	// Depth Texture For
	glGenTextures(1, &shadowMapCubeDepth);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowMapCubeDepth);
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_DEPTH_COMPONENT32F, 
				   LightDrawBuffer::ShadowMapWH, 
				   LightDrawBuffer::ShadowMapWH);

	// Interpret CubemapTexture Array as 2D Texture
	// Used for Directional Lights (each cube side is a cascade)
	glGenTextures(1, &shadowMapArrayView);
	glTextureView(shadowMapArrayView, GL_TEXTURE_2D_ARRAY, lightShadowMaps, GL_R32F,
				  0, LightDrawBuffer::ShadowMapMipCount, 0, CubeSide * lightCount);

	for(GLsizei i = 0; i < lightCount; i++)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBOs[i]);
		glTextureView(shadowMapViews[i], GL_TEXTURE_CUBE_MAP, lightShadowMaps, GL_R32F,
					  0, 1, 6 * i, 6);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, shadowMapViews[i], 0);
		glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, shadowMapCubeDepth, 0);
		assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
	}		
	lightsGPU.SendData();
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

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

const std::vector<IEMatrix4x4>&	SceneLights::getLightProjMatrices()
{
	return lightProjMatrices;
}

const std::vector<IEMatrix4x4>& SceneLights::getLightInvViewProjMatrices()
{
	return lightInvViewProjMatrices;
}

void SceneLights::ChangeLightPos(uint32_t index, IEVector3 position)
{
	lights[index].position[0] = position[0];
	lights[index].position[1] = position[1];
	lights[index].position[2] = position[2];
}

void SceneLights::ChangeLightDir(uint32_t index, IEVector3 direction)
{
	lights[index].direction[0] = direction[0];
	lights[index].direction[1] = direction[1];
	lights[index].direction[2] = direction[2];
}

void SceneLights::ChangeLightColor(uint32_t index, IEVector3 color)
{
	lights[index].color[0] = color[0];
	lights[index].color[1] = color[1];
	lights[index].color[2] = color[2];
}

void SceneLights::ChangeLightRadius(uint32_t index, float radius)
{
	lights[index].direction[3] = radius;
}

void SceneLights::ChangeLightIntensity(uint32_t index, float intensity)
{
	lights[index].color[3] = intensity;
}

void SceneLights::ChangeLightShadow(uint32_t index, bool shadowStatus)
{
	lightShadowCast[index] = shadowStatus;
}

IEVector3 SceneLights::getLightPos(uint32_t index) const
{
	return lights[index].position;
}

LightType SceneLights::getLightType(uint32_t index) const
{
	return static_cast<LightType>(static_cast<int>(lights[index].position.getW()));
}

IEVector3 SceneLights::getLightDir(uint32_t index) const
{
	return lights[index].direction;
}

IEVector3 SceneLights::getLightColor(uint32_t index) const
{
	return lights[index].color;
}

float SceneLights::getLightRadius(uint32_t index) const
{
	return lights[index].direction.getW();
}

float SceneLights::getLightIntensity(uint32_t index) const
{
	return lights[index].color.getW();
}

bool SceneLights::getLightCastShadow(uint32_t index) const
{
	return lightShadowCast[index];
}