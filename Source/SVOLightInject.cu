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

__device__ inline float3 DepthToWorld(volatile float2& uv,
									  const CMatrix4x4& projection,
									  const CMatrix4x4& invViewProjection,
									  cudaTextureObject_t shadowMaps,
									  const unsigned int lightMapIndex,
									  const float depthNear,
									  const float depthFar)
{

	// Converts Depthbuffer Value to World Coords
	// First Depthbuffer to Screen Space
	float shadowDepth = tex2DLayeredLod<float>(shadowMaps,
											   uv.x,
											   uv.y,
											   lightMapIndex,
											   0.0f);

	if(shadowDepth == 1.0f) return float3{INFINITY, INFINITY, INFINITY};

	// Normalize Device Coordinates
	float3 ndc =
	{
		2.0f * uv.x - 1.0f,
		2.0f * uv.y - 1.0f,
		2.0f * shadowDepth - 1.0f
	//	0.5f// ((2.0f * (shadowDepth - depthNear) / (depthFar - depthNear)) - 1.0f)
	};

	// Clip Space
	float3 clip;
	clip.z = ndc.z;
	clip.y = ndc.y;
	clip.x = ndc.x;
	
	// From Clip Space to World Space
	MultMatrixSelf(clip, invViewProjection);
	return {clip.x, clip.y, clip.z};
}


__device__ inline float3 PhongBRDF(const float3& worldPos,
								   const float4& camPos,
								   const float3& camDir,

								   // Light Related
								   const CLight& lightStruct,

								   // SVO Surface Voxel
								   const float4& colorSVO,
								   const float4& normalSVO)
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

	// Specular
	// Blinn-Phong
	float specPower = colorSVO.w * 4096.0f;
	float power = pow(fmaxf(Dot(worldHalf, worldNormal), 0.0f), specPower);
	lightIntensity.x += power;
	lightIntensity.y += power;
	lightIntensity.z += power;

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

//inline __device__ unsigned int AtomicAllocateNode(CSVONode* gNode, unsigned int& gLevelAllocator)
//{
//	// Release Configuration Optimization fucks up the code
//	// Prob changes some memory i-o ordering
//	// Its fixed but comment is here for future
//	// Problem here was cople threads on the same warp waits eachother and
//	// after some memory ordering changes by compiler responsible thread waits
//	// other threads execution to be done
//	// Code becomes something like this after compiler changes some memory orderings
//	//
//	//	while(old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE) == 0xFFFFFFFE); <-- notice semicolon
//	//	 if(old == 0xFFFFFF)
//	//		location = allocate();
//	//	location = old;
//	//	return location;
//	//
//	// first allocating thread will never return from that loop since 
//	// its warp threads are on infinite loop (so deadlock)
//
//	// much cooler version can be warp level exchange intrinsics
//	// which slightly reduces atomic pressure on the single node (on lower tree levels atleast)
//	if(*gNode < 0xFFFFFFFE) return *gNode;
//
//	CSVONode old = 0xFFFFFFFE;
//	while(old == 0xFFFFFFFE)
//	{
//		old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE);
//		if(old == 0xFFFFFFFF)
//		{
//			// Allocate
//			unsigned int location = atomicAdd(&gLevelAllocator, 8);
//			*reinterpret_cast<volatile CSVONode*>(gNode) = location;
//			old = location;
//		}
//		__threadfence();	// This is important somehow compiler changes this and makes infinite loop on same warp threads
//	}
//	return old;
//}

__global__ void SVOLightInject(// SVO Related
							   CSVOMaterial* gSVOMat,
							   const CSVONode* gSVOSparse,
							   const CSVONode* gSVODense,
							   const unsigned int* gLevelAllocators,

							   const unsigned int* gLevelOffsets,
							   const unsigned int* gLevelTotalSizes,

							   const CLight& gLightStruct,
							   const CSVOConstants& svoConstants,

							   const unsigned int matSparseOffset,

							   // Light Inject Related
							   float span,
							   const float3 outerCascadePos,

							   const float4 camPos,
							   const float3 camDir,

							   const float depthNear,
							   const float depthFar,

							   cudaTextureObject_t shadowMaps,

							   const CMatrix4x4 dLightProjection,
							   const CMatrix4x4 dLightInvViewProjection,

							   const unsigned int lightMapIndex,
							   const unsigned int mapWH)
{
	// For each pixel in the shadow map
	uint2 globalId =
	{
		threadIdx.x + blockIdx.x * blockDim.x,
		threadIdx.y + blockIdx.y * blockDim.y,
	};

	float2 shadowUV =
	{
		static_cast<float>(globalId.x) / static_cast<float>(mapWH),
		static_cast<float>(globalId.y) / static_cast<float>(mapWH)
	};

	float3 worldPoint = DepthToWorld(shadowUV,
									 dLightProjection,
									 dLightInvViewProjection,
									 shadowMaps,
									 lightMapIndex,
									 depthNear,
									 depthFar);

	// Convert to voxel integer space
	int3 voxPos =
	{
		static_cast<int>(floorf((worldPoint.x - outerCascadePos.x) / span)),
		static_cast<int>(floorf((worldPoint.y - outerCascadePos.y) / span)),
		static_cast<int>(floorf((worldPoint.z - outerCascadePos.z) / span))
	};

	// Cull if out of bounds
	if(voxPos.x < 0 || voxPos.x >= (0x1 << svoConstants.totalDepth) ||
	   voxPos.y < 0 || voxPos.y >= (0x1 << svoConstants.totalDepth) ||
	   voxPos.z < 0 || voxPos.z >= (0x1 << svoConstants.totalDepth))
	{
		return;
	}

    uint3 voxPosUnsigned =
    {
        static_cast<unsigned int>(voxPos.x),
        static_cast<unsigned int>(voxPos.y),
        static_cast<unsigned int>(voxPos.z)
	};

	// Traverse SVO
	uint3 levelVoxId = CalculateLevelVoxId(voxPosUnsigned, svoConstants.denseDepth, svoConstants.totalDepth);
	CSVONode nodeIndex = gSVODense[svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
								  svoConstants.denseDim * levelVoxId.y +
								  levelVoxId.x];
	if(nodeIndex == 0xFFFFFFFF) return;
	nodeIndex += CalculateLevelChildId(voxPosUnsigned, svoConstants.denseDepth + 1, svoConstants.totalDepth);

	unsigned int traverseLevel;
	for(traverseLevel = svoConstants.denseDepth + 1; 
		traverseLevel < svoConstants.totalDepth; 
		traverseLevel++)
	{
		unsigned int levelIndex = traverseLevel - svoConstants.denseDepth;
		const CSVONode currentNode = gSVOSparse[gLevelOffsets[levelIndex] + nodeIndex];
		if(currentNode == 0xFFFFFFFF) break;
		// Offset child
		unsigned int childId = CalculateLevelChildId(voxPosUnsigned, traverseLevel + 1, svoConstants.totalDepth);
		nodeIndex = currentNode + childId;
	}
	if(traverseLevel == svoConstants.totalDepth ||
	   traverseLevel > (svoConstants.totalDepth - svoConstants.numCascades))
	{
		// Write data to Location
        CSVOMaterial mat = gSVOMat[matSparseOffset + gLevelOffsets[traverseLevel - svoConstants.denseDepth] + nodeIndex];
	}

    // Debug SVO reconst 
	//unsigned int location;
    //unsigned int cascadeMaxLevel = svoConstants.totalDepth - 1;// (svoConstants.numCascades - 0);
	//for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
	//{
	//	unsigned int levelIndex = i - svoConstants.denseDepth;
	//	CSVONode* node = nullptr;
	//	if(i == svoConstants.denseDepth)
	//	{
	//		uint3 levelVoxId = CalculateLevelVoxId(voxPosUnsigned, i, svoConstants.totalDepth);
	//		node = gSVODense +
	//			svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
	//			svoConstants.denseDim * levelVoxId.y +
	//			levelVoxId.x;
	//	}
	//	else
	//	{
	//		node = gSVOSparse + gLevelOffsets[levelIndex] + location;
	//	}

	//	// Allocate (or acquire) next location
	//	location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1]);
	//	assert(location < gLevelTotalSizes[levelIndex + 1]);
	//	if(location >= gLevelTotalSizes[levelIndex + 1]) return;

	//	// Offset child
	//	unsigned int childId = CalculateLevelChildId(voxPosUnsigned, i + 1, svoConstants.totalDepth);
	//	location += childId;
	//}

 //   gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] + location] = {0xFFFFFFFFFFFFFFFF};
}
#endif //__SVOLIGHTINJECT_H__