#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"
#include "COpenGLTypes.h"
#include "CMatrixFunctions.cuh"

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f

#define GI_ONE_OVER_PI 0.318309f

inline __device__ float Dot(const float3& vec1, const float3& vec2)
{
	return	vec1.x * vec2.x +
			vec1.y * vec2.y +
			vec1.z * vec2.z;
}

//inline __device__ float Dot(const float4& vec1, const float4& vec2)
//{
//	return	vec1.x * vec2.x +
//			vec1.y * vec2.y +
//			vec1.z * vec2.z +
//			vec1.w * vec2.w;
//}

inline __device__ float Length(const float3& vec)
{
	return sqrtf(Dot(vec, vec));
}

//inline __device__ float Length(const float4& vec)
//{
//	return sqrtf(Dot(vec, vec));
//}

inline __device__ float3 Normalize(const float3& vec)
{
	float length = Length(vec);
	if(length != 0.0f) length = 1.0f / length;

	return float3
	{
		vec.x * length,
		vec.y * length,
		vec.z * length
	};
}

//inline __device__ float4 Normalize(const float4& vec)
//{
//	float length = Length(vec);
//	if(length != 0.0f) length = 1.0f / length;
//
//	return float4
//	{
//		vec.x * length,
//		vec.y * length,
//		vec.z * length,
//		vec.w * length
//	};
//}

inline __device__ float Clamp(float x, float min, float max)
{
	return  (x < min) ? min : ((x > max) ? max : x);
}

inline __device__ bool HasShadowOcclusionPoint(const CMatrix4x4* lightVP,
											   const CLight& lightStruct,
											   const float3& worldPos,
											   const CLightInjectParameters& liParams,
											   int lightIndex)
{
	//if(lightStruct.position.w == GI_LIGHT_POINT)
	//{
	//	// Determine which side of the light is the point
	//	// minimum absolute value
	//	lightVec.x = worldPos.x - lightStruct.position.x;
	//	lightVec.y = worldPos.y - lightStruct.position.y;
	//	lightVec.z = worldPos.z - lightStruct.position.z;

	//	float maxVal = fmaxf(abs(lightVec.x), fmaxf(abs(lightVec.y), abs(lightVec.z)));
	//	float3 axis;
	//	axis.x = (abs(lightVec.x) == maxVal) ? 1.0f : 0.0f;
	//	axis.y = (abs(lightVec.y) == maxVal) ? 1.0f : 0.0f;
	//	axis.z = (abs(lightVec.z) == maxVal) ? 1.0f : 0.0f;

	//	float3 lightVecSigns;
	//	lightVecSigns.x = ((lightVec.x * axis.x) < 0.0f) ? -1.0f : 1.0f;
	//	lightVecSigns.y = ((lightVec.y * axis.y) < 0.0f) ? -1.0f : 1.0f;
	//	lightVecSigns.z = ((lightVec.z * axis.z) < 0.0f) ? -1.0f : 1.0f;

	//	lightVecSigns.x = abs(lightVecSigns.x) * (abs((lightVecSigns.x - 1.0f) * 0.5f) + 0.0f);
	//	lightVecSigns.y = abs(lightVecSigns.y) * (abs((lightVecSigns.x - 1.0f) * 0.5f) + 2.0f);
	//	lightVecSigns.z = abs(lightVecSigns.z) * (abs((lightVecSigns.x - 1.0f) * 0.5f) + 4.0f);

	//	viewIndex = lightVecSigns.x + lightVecSigns.y + lightVecSigns.z;
	//}
	//else
		
	//{
	//	//// Cube Fetch if applicable
	//	//shadowDepth = texCubemapLayeredLod<float>(liParams.shadowMaps,
	//	//										  lightVec.x,
	//	//										  lightVec.y,
	//	//										  lightVec.z,
	//	//										  lightIndex,
	//	//										  0.0f);
	//}
	return true;
}

inline __device__ bool HasShadowOcclusionDirectional(const CMatrix4x4* lightVP,
													 const CLight& lightStruct,
													 const float3& worldPos,
													 const CLightInjectParameters& liParams,
													 int lightIndex)
{
	int viewIndex = 0;

	// Determine Cascade
	float3 distVec;
	distVec.x = worldPos.x - liParams.camPos.x;
	distVec.y = worldPos.y - liParams.camPos.y;
	distVec.z = worldPos.z - liParams.camPos.z;
	float worldDist = fmaxf(0.0f, Dot(distVec, liParams.camDir));

	// Inv geom sum
	const float exponent = 1.1f;
	float index = worldDist / liParams.camPos.w;
	index = round(log2(index * (exponent - 1.0f) + 1.0f) / log2(exponent));
	viewIndex = static_cast<int>(index);

	// Mult with proper cube side matrix
	float4 wPos = {worldPos.x, worldPos.y, worldPos.z, 1.0f};
	float4 clip = MultMatrix(wPos, lightVP[viewIndex]);

	// Convert to NDC
	float3 ndc;
	ndc.x = clip.x / clip.w;
	ndc.y = clip.y / clip.w;
	ndc.z = clip.z / clip.w;

	// NDC to Tex
	float depth = 0.5 * ((2.0f * liParams.depthNear + 1.0f) + 
						 (liParams.depthFar - liParams.depthNear) * ndc.z);

	float2 texUV;
	texUV.x = 0.5f * ndc.x + 0.5f;
	texUV.y = 0.5f * ndc.y + 0.5f;

	//texUV.x = 0.5f;
	//texUV.y = 0.5f;

	// Texture Array Fetch
	float shadowDepth = tex2DLayeredLod<float>(liParams.shadowMaps,
											   texUV.x,
											   texUV.y,
											   lightIndex * 6 + viewIndex,
											   0.0f);

	//if(blockIdx.x == 12 && threadIdx.x == 0)
	//	printf("world{%f %f %f}, uv{%f, %f} %d, shadowDepth %f \n",
	//		   worldPos.x, worldPos.y, worldPos.z,
	//		   lightVec.x, lightVec.y, viewIndex,
	//		   shadowDepth);

	// Cull if occluded
	if(shadowDepth < depth) return true;
	return false;
}

inline __device__ float3 PhongBRDFPoint(// Out Light
										float3& lightDirection,
										// Node Params
										const float3& worldPos,
										const float3& normal,
										const float& specularity,
										// Constants for this light
										const CLightInjectParameters& liParams,
										const CLight& lightStruct,
										const CMatrix4x4* lightVP,
										int lightIndex)
{
	//if(lightStruct.position.w == GI_LIGHT_DIRECTIONAL)
	//{
	//	worldLight.x = -lightStruct.direction.x;
	//	worldLight.y = -lightStruct.direction.y;
	//	worldLight.z = -lightStruct.direction.z;
	//}
	//else
	//{
	//	worldLight.x = lightStruct.position.x - worldPos.x;
	//	worldLight.y = lightStruct.position.y - worldPos.y;
	//	worldLight.z = lightStruct.position.z - worldPos.z;

	//	// Falloff Linear
	//	float lightRadius = lightStruct.direction.w;
	//	float distSqr = worldLight.x * worldLight.x +
	//					worldLight.y * worldLight.y +
	//					worldLight.z * worldLight.z;

	//	// Quadratic Falloff
	//	falloff = distSqr / (lightRadius * lightRadius);
	//	falloff = Clamp(1.0f - falloff * falloff, 0.0f, 1.0f);
	//	falloff = (falloff * falloff) / (distSqr + 1.0f);
	//}

	//if(lightStruct.position.w == GI_LIGHT_POINT)
	//{
	//	lightDirection.x *= falloff;
	//	lightDirection.y *= falloff;
	//	lightDirection.z *= falloff;
	//}
	return float3{0.0f, 0.0f, 0.0f};
}

inline __device__ float3 PhongBRDFDirectional(// Out Light
											  float3& lightDirection,
											  // Node Params
											  const float3& worldPos,
											  const float3& normal,
											  const float& specularity,
											  // Constants for this light
											  const CLightInjectParameters& liParams,
											  const CLight& lightStruct,
											  const CMatrix4x4* lightVP,
											  int lightIndex)

{
	float3 irradiance = {0.0f, 0.0f, 0.0f};
	lightDirection = {0.0f, 0.0f, 0.0f};

	float3 worldEye;
	worldEye.x = liParams.camPos.x - worldPos.x;
	worldEye.y = liParams.camPos.y - worldPos.y;
	worldEye.z = liParams.camPos.z - worldPos.z;
	worldEye = Normalize(worldEye);

	float3 worldLight;
	worldLight.x = -lightStruct.direction.x;
	worldLight.y = -lightStruct.direction.y;
	worldLight.z = -lightStruct.direction.z;
	worldLight = Normalize(worldLight);
		
	// Early bail if back surface
	float NdL = fmaxf(Dot(normal, worldLight), 0.0f);
	//if(NdL <= 0.0f) return irradiance;
	
	// Light Occlusion Check
	// Bias the world a little bit since voxel system and world system do not align perfectly
	// Only for shadow map fetch
	float3 biasedWorld = worldPos;
	biasedWorld.x += worldLight.x * 1.5f;
	biasedWorld.y += worldLight.y * 1.5f;
	biasedWorld.z += worldLight.z * 1.5f;

	bool occluded = HasShadowOcclusionDirectional(lightVP,
												  lightStruct,
												  biasedWorld,
												  liParams,
												  lightIndex);
	if(occluded) return irradiance;
	
	// Lambert Diffuse Model
	irradiance = {GI_ONE_OVER_PI, GI_ONE_OVER_PI, GI_ONE_OVER_PI};

	// Specular
	float3 worldHalf;
	worldHalf.x = worldLight.x + worldEye.x;
	worldHalf.y = worldLight.y + worldEye.y;
	worldHalf.z = worldLight.z + worldEye.z;
	worldHalf = Normalize(worldHalf);

	// Blinn-Phong
	float specPower = 16.0f + specularity * 2048.0f;
	float power = GI_ONE_OVER_PI * 0.125f * (specPower + 6.0f) * 
				  (pow(fmaxf(Dot(worldHalf, normal), 0.0f), specPower)) * 900.0f;
	irradiance.x += power;
	irradiance.y += power;
	irradiance.z += power;

	// NdL + Colorize + Intensity + Falloff
	irradiance.x *= NdL * lightStruct.color.x * lightStruct.color.w;
	irradiance.y *= NdL * lightStruct.color.y * lightStruct.color.w;
	irradiance.z *= NdL * lightStruct.color.z * lightStruct.color.w;

	lightDirection = worldLight;
	return irradiance;
}

inline __device__ float3 TotalIrradiance(float3& mainLightDir,
										 // Node Params
										 const float3& worldPos,
										 const float3& normal,
										 const float4& albedo,
										 // Light Parameters
										 const CLightInjectParameters& liParams)
{
	float3 totalIllum = liParams.ambientLight;

	//for(int lightId = 0; lightId < liParams.lightCount; lightId++)
	int lightId = 0;
	{
		const CLight& light = liParams.gLightStruct[lightId];
		const CMatrix4x4* lightVP = liParams.gLightVP + (lightId * 6);
		
		float3 lightDir;
		float3 illum = PhongBRDFDirectional(// Out Light
											lightDir,
											// Node Params
											worldPos,
											normal,
											albedo.w,
											// Constants for this light
											liParams,
											light,
											lightVP,
											lightId);

		totalIllum.x += illum.x;
		totalIllum.y += illum.y;
		totalIllum.z += illum.z;

		mainLightDir.x += lightDir.x;
		mainLightDir.y += lightDir.y;
		mainLightDir.z += lightDir.z;
	}
	
	float invCoutner = 1.0f / static_cast<float>(liParams.lightCount);
	mainLightDir.x *= invCoutner;
	mainLightDir.y *= invCoutner;
	mainLightDir.z *= invCoutner;

	totalIllum.x = fminf(totalIllum.x, 1.0f);
	totalIllum.y = fminf(totalIllum.y, 1.0f);
	totalIllum.z = fminf(totalIllum.z, 1.0f);
	totalIllum.x *= albedo.x;
	totalIllum.y *= albedo.y;
	totalIllum.z *= albedo.z;
	return totalIllum;
}