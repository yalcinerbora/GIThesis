#ifndef __SVOLIGHTINJECT_H__
#define __SVOLIGHTINJECT_H__

#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CVoxel.cuh"

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f
#define GI_LIGHT_AREA 2.0f

#define GI_ONE_OVER_PI 0.318309f

__device__ inline float Dot(const float3& vec1, const float3& vec2)
{
	return	vec1.x * vec2.x +
			vec1.y * vec2.y +
			vec1.z * vec2.z;
}

__device__ inline float Dot(const float4& vec1, const float4& vec2)
{
	return	vec1.x * vec2.x +
			vec1.y * vec2.y +
			vec1.z * vec2.z +
			vec1.w * vec2.w;
}

__device__ inline float Length(const float3& vec)
{
	return sqrtf(Dot(vec, vec));
}

__device__ inline float Length(const float4& vec)
{
	return sqrtf(Dot(vec, vec));
}

__device__ inline float3 Normalize(const float3& vec)
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

__device__ inline float4 Normalize(const float4& vec)
{
	float length = Length(vec);
	if(length != 0.0f) length = 1.0f / length;

	return float4
	{
		vec.x * length,
		vec.y * length,
		vec.z * length,
		vec.w * length
	};
}

__device__ inline float Clamp(float x, float min, float max)
{
	return  (x < min) ? min : ((x > max) ? max : x);
}

__device__ inline float4 CalculateShadowUV(const CMatrix4x4* lightVP,
										   const CLight& lightStruct,
										   const float3& worldPos,
										   const float4& camPos,
										   const float3& camDir,
										   float depthNear,
										   float depthFar)
{
	float viewIndex = 0.0f;
	float3 lightVec;
	if(lightStruct.position.w == GI_LIGHT_POINT ||
	   lightStruct.position.w == GI_LIGHT_AREA)
	{
		// Determine which side of the light is the point
		// minimum absolute value
		lightVec.x = worldPos.x - lightStruct.position.x;
		lightVec.y = worldPos.y - lightStruct.position.y;
		lightVec.z = worldPos.z - lightStruct.position.z;

		float maxVal = fmaxf(abs(lightVec.x), fmaxf(abs(lightVec.y), abs(lightVec.z)));
		float3 axis;
		axis.x = (abs(lightVec.x) == maxVal) ? 1.0f : 0.0f;
		axis.y = (abs(lightVec.y) == maxVal) ? 1.0f : 0.0f;
		axis.z = (abs(lightVec.z) == maxVal) ? 1.0f : 0.0f;

		float3 lightVecSigns;
		lightVecSigns.x = ((lightVec.x * axis.x) < 0.0f) ? -1.0f : 1.0f;
		lightVecSigns.y = ((lightVec.y * axis.y) < 0.0f) ? -1.0f : 1.0f;
		lightVecSigns.z = ((lightVec.z * axis.z) < 0.0f) ? -1.0f : 1.0f;

		lightVecSigns.x = abs(lightVecSigns.x) * (abs((lightVecSigns.x - 1.0f) * 0.5f) + 0.0f);
		lightVecSigns.y = abs(lightVecSigns.y) * (abs((lightVecSigns.x - 1.0f) * 0.5f) + 2.0f);
		lightVecSigns.z = abs(lightVecSigns.z) * (abs((lightVecSigns.x - 1.0f) * 0.5f) + 4.0f);

		viewIndex = lightVecSigns.x + lightVecSigns.y + lightVecSigns.z;

		// Area light is half sphere
		if(lightStruct.position.w == GI_LIGHT_AREA)
			viewIndex = (lightVec.y < 0.0f) ? viewIndex : 2.0f;
	}
	else
	{
		// Determine Cascade
		float3 distVec;
		distVec.x = worldPos.x - camPos.x;
		distVec.y = worldPos.y - camPos.y;
		distVec.z = worldPos.z - camPos.z;
		
		float worldDist = fmaxf(0.0f, Dot(distVec, camDir));

		// Inv geom sum
		const float exponent = 1.2f;
		viewIndex = worldDist / camPos.w;
		viewIndex = floor(log2(viewIndex * (exponent - 1.0f) + 1.0f) / log2(exponent));
	}

	// Mult with proper cube side matrix
	float4 wPos = {worldPos.x, worldPos.y, worldPos.z, 1.0f};
	float4 clip = MultMatrix(wPos, lightVP[static_cast<int>(viewIndex)]);

	// Convert to NDC
	float3 ndc;
	ndc.x = clip.x / clip.w;
	ndc.y = clip.y / clip.w;
	ndc.z = clip.z / clip.w;

	// NDC to Tex
	float depth = 0.5 * ((2.0f * depthNear + 1.0f) + 
						 (depthFar - depthNear) * ndc.z);
	if(lightStruct.position.w == GI_LIGHT_DIRECTIONAL)
	{
		lightVec.x = 0.5f * ndc.x + 0.5f;
		lightVec.y = 0.5f * ndc.y + 0.5f;
		lightVec.z = viewIndex;
	}
	return float4{lightVec.x, lightVec.y, lightVec.z, depth};
}

__device__ inline float3 PhongBRDF(const float3& worldPos,
								   const float4& camPos,
								   const float3& camDir,

								   // Light Related
								   const CMatrix4x4* lightVP,
								   const CLight& lightStruct,

								   float depthNear,
								   float depthFar,

								   // SVO Surface Voxel
								   const float4& colorSVO,
								   const float4& normalSVO,

								   cudaTextureObject_t shadowTex,
								   uint32_t lightIndex)
{
	float3 lightIntensity = {0.0f, 0.0f, 0.0f};

	float3 worldEye;
	worldEye.x = camPos.x - worldPos.x;
	worldEye.y = camPos.y - worldPos.y;
	worldEye.z = camPos.z - worldPos.z;

	float3 worldLight;
	float falloff = 1.0f;
	if(lightStruct.position.w == GI_LIGHT_DIRECTIONAL)
	{
		worldLight.x = -lightStruct.direction.x;
		worldLight.y = -lightStruct.direction.y;
		worldLight.z = -lightStruct.direction.z;
	}
	else
	{
		worldLight.x = lightStruct.position.x - worldPos.x;
		worldLight.y = lightStruct.position.y - worldPos.y;
		worldLight.z = lightStruct.position.z - worldPos.z;

		// Falloff Linear
		float lightRadius = lightStruct.direction.w;
		float distSqr = worldLight.x * worldLight.x +
						worldLight.y * worldLight.y +
						worldLight.z * worldLight.z;

		// Quadratic Falloff
		falloff = distSqr / (lightRadius * lightRadius);
		falloff = Clamp(1.0f - falloff * falloff, 0.0f, 1.0f);
		falloff = (falloff * falloff) / (distSqr + 1.0f);
	}
	worldLight = Normalize(worldLight);
	worldEye = Normalize(worldEye);
	
	float3 biasedWorld = worldPos;
	biasedWorld.x += worldLight.x * 3.0f;
	biasedWorld.y += worldLight.y * 3.0f;
	biasedWorld.z += worldLight.z * 3.0f;

	float4 shadowUV = CalculateShadowUV(lightVP,
										lightStruct,
										biasedWorld,
										camPos,
										camDir,
										depthNear,
										depthFar);



	float3 worldHalf;
	worldHalf.x = worldLight.x + worldEye.x;
	worldHalf.y = worldLight.y + worldEye.y;
	worldHalf.z = worldLight.z + worldEye.z;
	worldHalf = Normalize(worldHalf);

	float3 worldNormal = {normalSVO.x, normalSVO.y, normalSVO.z};

	// Lambert Diffuse Model
	float intensity = fmaxf(Dot(worldNormal, worldLight), 0.0f) * GI_ONE_OVER_PI;
	lightIntensity = {intensity, intensity, intensity};

	// Early Bail From Light Occulusion
	// This also eliminates some self shadowing artifacts
	if(intensity == 0.0f) return lightIntensity;

	// Check Light Occulusion (ShadowMap)
	float shadowDepth = 0.0f;
	if(lightStruct.position.w == GI_LIGHT_DIRECTIONAL)
	{
		// Texture Array Fetch
		shadowDepth = tex2DLayeredLod<float>(shadowTex,
											 shadowUV.x,
											 shadowUV.y,
											 static_cast<float>(lightIndex * 6) + shadowUV.z,
											 0.0f);
	}
	else
	{
		// Cube Fetch if applicable
		shadowDepth = texCubemapLayeredLod<float>(shadowTex,
												  shadowUV.x,
												  shadowUV.y,
												  shadowUV.z,
												  lightIndex,
												  0.0f);
	}

	// Cull if occluded
	if(shadowDepth < shadowUV.w) return float3{0.0f, 0.0f, 0.0f};

	// Specular
	// Blinn-Phong
	//float specPower = colorSVO.w * 4096.0f;
	//float power = pow(fmaxf(Dot(worldHalf, worldNormal), 0.0f), specPower);
	//lightIntensity.x += power;
	//lightIntensity.y += power;
	//lightIntensity.z += power;

	// Falloff
	lightIntensity.x *= falloff;
	lightIntensity.y *= falloff;
	lightIntensity.z *= falloff;

	// Colorize + Intensity
	lightIntensity.x *= lightStruct.color.x * lightStruct.color.w;
	lightIntensity.y *= lightStruct.color.y * lightStruct.color.w;
	lightIntensity.z *= lightStruct.color.z * lightStruct.color.w;

	// Out
	float3 result =
	{
		lightIntensity.x * colorSVO.x,
		lightIntensity.y * colorSVO.y,
		lightIntensity.z * colorSVO.z
	};

	result.x = Clamp(result.x, 0.0f, 1.0f);
	result.y = Clamp(result.y, 0.0f, 1.0f);
	result.z = Clamp(result.z, 0.0f, 1.0f);
	return result;
}

__device__ inline float3 LightInject(const float3& worldPos,

									 // SVO Surface Voxel
									 const float4& colorSVO,
									 const float4& normalSVO,

									 // Camera Related
									 const float4& camPos,
									 const float3& camDir,

									 // Light View Projection
									 const CMatrix4x4* lightVP,
									 const CLight* lightStruct,

									 float depthNear,
									 float depthFar,

									 // Shadow Tex
									 cudaTextureObject_t shadowTex,

									 const int lightCount)
{
	// For Each Light
	float3 totalIllum = {0.0f, 0.0f, 0.0f};
	for(int i = 0; i < lightCount; i++)
	{
		const CMatrix4x4* currentLightVP = lightVP + (i * 6);
		const CLight light = lightStruct[i];

		float3 illum = PhongBRDF(worldPos,
								 camPos,
								 camDir,
								 currentLightVP,
								 light,
								 depthNear,
								 depthFar,
								 colorSVO,
								 normalSVO,
								 shadowTex,
								 i);

		totalIllum.x += illum.x;
		totalIllum.y += illum.y;
		totalIllum.z += illum.z;
	}
	return totalIllum;
}

__global__ void LightInject(CSVOMaterial* gSVOMat,
							const CSVONode* gSVOSparse,
							const CSVONode* gSVODense,
							const unsigned int* gLevelAllocators,

							const unsigned int* gLevelOffsets,
							const unsigned int* gLevelTotalSizes,

							// Light Inject Related
							float span,
							const float3 outerCascadePos,

							const float4 camPos,
							const float3 camDir,

							const CMatrix4x4* lightVP,
							const CLight* lightStruct,

							const float depthNear,
							const float depthFar,

							cudaTextureObject_t shadowMaps,
							const unsigned int lightCount,
							const unsigned int mapWH)
{
	//// For each pixel in the shadow map
	//uint2 globalId = 
	//{
	//	threadIdx.x + blockIdx.x * blockDim.x,
	//	threadIdx.y + blockIdx.y * blockDim.y, 
	//};

	//globalId

	//float viewIndex = 0.0f;
	//float3 lightVec;
	//if(lightStruct.position.w == GI_LIGHT_POINT ||
	//   lightStruct.position.w == GI_LIGHT_AREA)
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

	//	// Area light is half sphere
	//	if(lightStruct.position.w == GI_LIGHT_AREA)
	//		viewIndex = (lightVec.y < 0.0f) ? viewIndex : 2.0f;
	//}
	//else
	//{
	//	// Determine Cascade
	//	float3 distVec;
	//	distVec.x = worldPos.x - camPos.x;
	//	distVec.y = worldPos.y - camPos.y;
	//	distVec.z = worldPos.z - camPos.z;

	//	float worldDist = fmaxf(0.0f, Dot(distVec, camDir));

	//	// Inv geom sum
	//	const float exponent = 1.2f;
	//	viewIndex = worldDist / camPos.w;
	//	viewIndex = floor(log2(viewIndex * (exponent - 1.0f) + 1.0f) / log2(exponent));
	//}

	//// Mult with proper cube side matrix
	//float4 wPos = {worldPos.x, worldPos.y, worldPos.z, 1.0f};
	//float4 clip = MultMatrix(wPos, lightVP[static_cast<int>(viewIndex)]);

	//// Convert to NDC
	//float3 ndc;
	//ndc.x = clip.x / clip.w;
	//ndc.y = clip.y / clip.w;
	//ndc.z = clip.z / clip.w;

	//// NDC to Tex
	//float depth = 0.5 * ((2.0f * depthNear + 1.0f) +
	//					 (depthFar - depthNear) * ndc.z);
	//if(lightStruct.position.w == GI_LIGHT_DIRECTIONAL)
	//{
	//	lightVec.x = 0.5f * ndc.x + 0.5f;
	//	lightVec.y = 0.5f * ndc.y + 0.5f;
	//	lightVec.z = viewIndex;
	//}
	//return float4{lightVec.x, lightVec.y, lightVec.z, depth};


	//

	//

	//globalId = 0;



	//globalId.


	//// Light Injection
	//if(inject)
	//{
	//	float4 colorSVO = UnpackSVOColor(voxelColorPacked);
	//	float4 normalSVO = ExpandOnlyNormal(voxelNormPacked);

	//	float3 worldPos =
	//	{
	//		outerCascadePos.x + voxelPos.x * span,
	//		outerCascadePos.y + voxelPos.y * span,
	//		outerCascadePos.z + voxelPos.z * span
	//	};

	//	// First Averager find and inject light
	//	float3 illum = LightInject(worldPos,

	//							   colorSVO,
	//							   normalSVO,

	//							   camPos,
	//							   camDir,

	//							   lightVP,
	//							   lightStruct,

	//							   depthNear,
	//							   depthFar,

	//							   shadowMaps,
	//							   lightCount);

	//	colorSVO.x = illum.x;
	//	colorSVO.y = illum.y;
	//	colorSVO.z = illum.z;
	//	voxelColorPacked = PackSVOColor(colorSVO);
	//}
}


#endif //__SVOLIGHTINJECT_H__