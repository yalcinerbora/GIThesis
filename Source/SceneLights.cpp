#include "SceneLights.h"
#include "IEUtility/IEVector3.h"
#include "VertexBuffer.h"
#include "DrawBuffer.h"
#include "Globals.h"
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
	static constexpr float exponent = 1.1f;
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
																  PointLightNear, currentLight.color.getW());
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
				float totalLength = 0.0f;
				for(int j = 0; j < LightDrawBuffer::DirectionalCascadesCount; j++)
				{
					float cascade = CalculateCascadeLength(camera.far, j);
					IEBoundingSphere viewSphere = CalculateShadowCascasde(totalLength,
																		  totalLength + cascade,
																		  camera,
																		  currentLight.direction);
					totalLength += cascade;

					// Squre Orto Projection
					float radius = viewSphere.radius;
					//float factor = static_cast<float>(2.0 / IEMathConstants::Sqrt2);
					IEMatrix4x4 projection = IEMatrix4x4::Ortogonal(2.0f * radius, 2.0f * radius, 
																	DirectionalLightNear, 
																	DirectionalLightFar);


					IEMatrix4x4 view = IEMatrix4x4::LookAt(viewSphere.center,
														   viewSphere.center + currentLight.direction,
														   camera.up);

					// To eliminate shadow shimmering only change pixel sized frusutm changes
					IEVector3 unitPerTexel = (2.0f * IEVector3(radius)) / 
											 IEVector3(static_cast<float>(LightDrawBuffer::ShadowMapWH));
					unitPerTexel *= static_cast<float>(1 << (LightDrawBuffer::ShadowMipSampleCount));
					IEVector3 translatedOrigin = view * IEVector3::ZeroVector;
					IEVector3 texelTranslate;
					texelTranslate.setX(std::fmod(translatedOrigin.getX(), unitPerTexel.getX()));
					texelTranslate.setY(std::fmod(translatedOrigin.getY(), unitPerTexel.getY()));
					texelTranslate.setZ(std::fmod(translatedOrigin.getZ(), unitPerTexel.getZ()));
					texelTranslate = unitPerTexel - texelTranslate;

					IEMatrix4x4 texelTranslateMatrix = IEMatrix4x4::Translate(texelTranslate);
					lightViewProjMatrices[i * 6 + j] = projection * texelTranslateMatrix * view;
					lightProjMatrices[i * 6 + j] = projection;
					lightInvViewProjMatrices[i * 6 + j] = (projection * texelTranslateMatrix * view).Inverse();
				}
				break;
			}
			//case LightType::RECTANGULAR:
			//{
			//	IEMatrix4x4 projections[2] = 
			//	{
			//		IEMatrix4x4::Perspective(45.0f, 1.0f, 0.1f, currentLight.color.getW()),
			//		IEMatrix4x4::Perspective(90.0f, 1.0f, 0.1f, currentLight.color.getW())
			//	};

			//	// we'll use 5 sides but each will comply different ares that a point light
			//	for(unsigned int j = 0; j < 6; j++)
			//	{
			//		uint32_t projIndex = (j == 3) ? 1 : 0;
			//		IEMatrix4x4 view = IEMatrix4x4::LookAt(currentLight.position,
			//											   currentLight.position + SceneLights::aLightDir[j],
			//											   SceneLights::aLightUp[j]);

			//		lightViewProjMatrices[i * 6 + j] = projections[projIndex] * view;
			//		lightProjMatrices[i * 6 + j] = projections[projIndex];
			//		lightInvViewProjMatrices[i * 6 + j] = (projections[projIndex] * view).Inverse();
			//	}
			//	break;
			//}
		}
	}
}

SceneLights::SceneLights()
	: lightOffset(0)
	, matrixOffset(0)
	, lightShadowMaps(0)
	, shadowMapArrayView(0)
	, shadowMapCubeDepth(0)
{
	for(auto& l : lightCounts) l = 0;
}

SceneLights::SceneLights(const std::vector<Light>& lights)
	: gpuData(lights.size() * (sizeof(Light) + sizeof(IEMatrix4x4) * CubeSide + sizeof(uint32_t)))
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
	for(auto& l : lightCounts) l = 0;
	GLsizei lightCount = static_cast<GLsizei>(lights.size());

	glGenTextures(1, &lightShadowMaps);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, lightShadowMaps);
	glTexStorage3D(GL_TEXTURE_CUBE_MAP_ARRAY, LightDrawBuffer::ShadowMapMipCount, 
				   GL_R32F, LightDrawBuffer::ShadowMapWH, LightDrawBuffer::ShadowMapWH,
				   CubeSide * lightCount);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_COMPARE_FUNC, GL_GREATER);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAX_LEVEL, LightDrawBuffer::ShadowMapMipCount);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_BASE_LEVEL, 0);
	
	glGenFramebuffers(lightCount, shadowMapFBOs.data());
	glGenTextures(lightCount, shadowMapViews.data());

	// Depth Texture For
	glGenTextures(1, &shadowMapCubeDepth);
	glBindTexture(GL_TEXTURE_CUBE_MAP, shadowMapCubeDepth);
	glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_DEPTH_COMPONENT32F, 
				   LightDrawBuffer::ShadowMapWH, 
				   LightDrawBuffer::ShadowMapWH);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, 0);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BASE_LEVEL, 0);

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
	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	//Generate Light Index Buffer
	int index = 0;
	std::array<std::vector<uint32_t>, LightTypeCount> lightTypeIndices;	
	for(const Light& l : lights)
	{
		int typeIndex = static_cast<int>(l.position.getW());
		lightTypeIndices[typeIndex].push_back(index);
		lightCounts[typeIndex]++;
		index++;
	}
	for(const auto& indices : lightTypeIndices)
	{
		lightIndices.insert(lightIndices.end(), indices.begin(), indices.end());
	}

	// Everything is Generated Now Construct Buffer
	size_t totalSize = 0;
	// LightStruct
	lightOffset = totalSize;
	totalSize += lights.size() * sizeof(Light);
	// ViewProjTransforms
	totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	matrixOffset = totalSize;
	totalSize += lightViewProjMatrices.size() * sizeof(IEMatrix4x4);
	// LightIndexOffset
	//totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	lightIndexOffset = totalSize;
	totalSize += lightIndices.size() * sizeof(uint32_t);

	// Copy To Buffer
	auto& cpuImage = gpuData.CPUData();
	cpuImage.resize(totalSize);
	std::copy(reinterpret_cast<const uint8_t*>(lights.data()),
			  reinterpret_cast<const uint8_t*>(lights.data() + lights.size()),
			  cpuImage.data() + lightOffset);
	std::copy(reinterpret_cast<uint8_t*>(lightViewProjMatrices.data()),
			  reinterpret_cast<uint8_t*>(lightViewProjMatrices.data() + lightViewProjMatrices.size()),
			  cpuImage.data() + matrixOffset);
	std::copy(reinterpret_cast<uint8_t*>(lightIndices.data()),
			  reinterpret_cast<uint8_t*>(lightIndices.data() + lightIndices.size()),
			  cpuImage.data() + lightIndexOffset);

	// Finalize and Send
	gpuData.SendData();
}

SceneLights::SceneLights(SceneLights&& other)
	: gpuData(std::move(other.gpuData))
	, lightOffset(other.lightOffset)
	, matrixOffset(other.matrixOffset)
	, lightIndexOffset(other.lightIndexOffset)
	, lightShadowMaps(other.lightShadowMaps)
	, shadowMapArrayView(other.shadowMapArrayView)
	, shadowMapCubeDepth(other.shadowMapCubeDepth)
	, shadowMapViews(std::move(other.shadowMapViews))
	, shadowMapFBOs(std::move(other.shadowMapFBOs))
	, lightCounts(std::move(other.lightCounts))
	, lightIndices(std::move(other.lightIndices))
	, lights(std::move(other.lights))
	, lightViewProjMatrices(std::move(other.lightViewProjMatrices))
	, lightProjMatrices(std::move(other.lightProjMatrices))
	, lightInvViewProjMatrices(std::move(other.lightInvViewProjMatrices))
	, lightShadowCast(std::move(other.lightShadowCast))
{
	other.lightShadowMaps = 0;
	other.shadowMapArrayView = 0;
	other.shadowMapCubeDepth = 0;
}

SceneLights& SceneLights::operator=(SceneLights&& other)
{
	assert(this != &other);
	if(shadowMapFBOs.size() > 0) glDeleteFramebuffers(static_cast<GLsizei>(shadowMapFBOs.size()),
													  shadowMapFBOs.data());
	if(shadowMapViews.size() > 0) glDeleteFramebuffers(static_cast<GLsizei>(shadowMapViews.size()),
													   shadowMapViews.data());
	glDeleteTextures(1, &shadowMapCubeDepth);
	glDeleteTextures(1, &lightShadowMaps);
	glDeleteTextures(1, &shadowMapArrayView);
	
	gpuData = std::move(other.gpuData);
	lightOffset = other.lightOffset;
	matrixOffset = other.matrixOffset;
	lightIndexOffset = other.lightIndexOffset;
	lightShadowMaps = other.lightShadowMaps;
	shadowMapArrayView = other.shadowMapArrayView;
	shadowMapCubeDepth = other.shadowMapCubeDepth;
	shadowMapViews = std::move(other.shadowMapViews);
	shadowMapFBOs = std::move(other.shadowMapFBOs);
	lightCounts = std::move(other.lightCounts);
	lightIndices = std::move(other.lightIndices);
	lights = std::move(other.lights);
	lightViewProjMatrices = std::move(other.lightViewProjMatrices);
	lightProjMatrices = std::move(other.lightProjMatrices);
	lightInvViewProjMatrices = std::move(other.lightInvViewProjMatrices);
	lightShadowCast = std::move(other.lightShadowCast);

	other.lightShadowMaps = 0;
	other.shadowMapArrayView = 0;
	other.shadowMapCubeDepth = 0;
	return *this;
}

SceneLights::~SceneLights()
{
	if(shadowMapFBOs.size() > 0) glDeleteFramebuffers(static_cast<GLsizei>(shadowMapFBOs.size()), 
													  shadowMapFBOs.data());
	if(shadowMapViews.size() > 0) glDeleteFramebuffers(static_cast<GLsizei>(shadowMapViews.size()), 
													   shadowMapViews.data());
	glDeleteTextures(1, &shadowMapCubeDepth);
	glDeleteTextures(1, &lightShadowMaps);
	glDeleteTextures(1, &shadowMapArrayView);
}

uint32_t SceneLights::getLightCount() const
{
	return static_cast<uint32_t>(lightShadowCast.size());
}

uint32_t SceneLights::getLightCount(LightType t) const
{
	return lightCounts[static_cast<int>(t)];
}

const std::vector<IEMatrix4x4>&	SceneLights::getLightProjMatrices() const
{
	return lightProjMatrices;
}

const std::vector<IEMatrix4x4>& SceneLights::getLightInvViewProjMatrices() const
{
	return lightInvViewProjMatrices;
}

float SceneLights::getCascadeLength(float cameraFar) const
{
	return CalculateCascadeLength(cameraFar, 0);
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

void SceneLights::BindLightFramebuffer(uint32_t light)
{
	glBindFramebuffer(GL_FRAMEBUFFER, shadowMapFBOs[light]);
}

void SceneLights::BindViewProjectionMatrices(GLuint bindPoint)
{
	gpuData.BindAsShaderStorageBuffer(bindPoint,
									  static_cast<GLuint>(matrixOffset),
									  static_cast<GLuint>(lightViewProjMatrices.size() * sizeof(IEMatrix4x4)));
}

void SceneLights::BindLightParameters(GLuint bindPoint)
{
	gpuData.BindAsShaderStorageBuffer(bindPoint,
									  static_cast<GLuint>(lightOffset),
									  static_cast<GLuint>(lights.size() * sizeof(Light)));
}

GLuint SceneLights::getGLBuffer()
{
	return gpuData.getGLBuffer();
}

GLuint SceneLights::getShadowTextureCubemapArray()
{
	return lightShadowMaps;
}

GLuint SceneLights::getShadowTextureArrayView()
{
	return shadowMapArrayView;
}

size_t SceneLights::getLightOffset() const
{
	return lightOffset;
}

size_t SceneLights::getLightIndexOffset() const
{
	return lightIndexOffset;
}

size_t SceneLights::getMatrixOffset() const
{
	return matrixOffset;
}

void SceneLights::SendVPMatricesToGPU()
{
	std::copy(reinterpret_cast<uint8_t*>(lightViewProjMatrices.data()),
			  reinterpret_cast<uint8_t*>(lightViewProjMatrices.data() + lightViewProjMatrices.size()),
			  gpuData.CPUData().data() + matrixOffset);
	gpuData.SendSubData(static_cast<uint32_t>(matrixOffset), 
						static_cast<uint32_t>(lightViewProjMatrices.size() * sizeof(IEMatrix4x4)));
}

void SceneLights::SendLightDataToGPU()
{
	std::copy(reinterpret_cast<uint8_t*>(lights.data()),
			  reinterpret_cast<uint8_t*>(lights.data() + lights.size()),
			  gpuData.CPUData().data() + lightOffset);
	gpuData.SendSubData(static_cast<uint32_t>(lightOffset),
						static_cast<uint32_t>(lights.size() * sizeof(Light)));
}