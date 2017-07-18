#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"


__device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
									const float4& upperPortion,
									const float2& lowerPortion,
									float occlusion);

__device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
									const float4& lowerPortion,
									const float occlusion);

__device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
										  const float4& upperPortion,
										  const float2& lowerPortion,
										  float occlusion);

__device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
										  const float4& upperPortion,
										  float occlusion);

inline __device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
										   const float4& lowerPortion,
										   const float occlusion)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(illumPortion);
	float4 avgLower = UnpackLightDirLeaf(illumSplit.x);
	float counter = uint_as_float(illumSplit.y);

	// Divisors
	float invCounter = 1.0f / (counter + occlusion);

	// Lower Average
	avgLower.x = (counter * avgLower.x + lowerPortion.x * occlusion) * invCounter;
	avgLower.y = (counter * avgLower.y + lowerPortion.y * occlusion) * invCounter;
	avgLower.z = (counter * avgLower.z + lowerPortion.z * occlusion) * invCounter;
	avgLower.w = (counter * avgLower.w + lowerPortion.w * occlusion) * invCounter;

	// Average
	counter += occlusion;
	illumSplit.y = __float_as_uint(counter);

	return PackWords(illumSplit.y, PackLightDirLeaf(avgLower));
}

inline __device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
										   const float4& lowerPortion,
										   const float2& upperPortion,
										   float occlusion)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(illumPortion);
	float4 avgLower = UnpackSVOIrradiance(illumSplit.x);
	float3 avgUpper = UnpackSVOUpperLeaf(illumSplit.y);
	
	// Divisors
	float counter = avgUpper.z;
	float invCounter = 1.0f / (avgUpper.z + occlusion);

	// Lower Average
	avgLower.x = (counter * avgLower.x + lowerPortion.x * occlusion) * invCounter;
	avgLower.y = (counter * avgLower.y + lowerPortion.y * occlusion) * invCounter;
	avgLower.z = (counter * avgLower.z + lowerPortion.z * occlusion) * invCounter;
	avgLower.w = (counter * avgLower.w + lowerPortion.w * occlusion) * invCounter;

	// Upper Average
	avgUpper.x = (counter * avgUpper.x + upperPortion.x * occlusion) * invCounter;
	avgUpper.y = (counter * avgUpper.y + upperPortion.y * occlusion) * invCounter;

	// Average
	avgUpper.z += occlusion;

	return PackWords(PackSVOUpperLeaf(avgUpper), PackSVOIrradiance(avgLower));
}

inline __device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
												 const float4& upperPortion,
												 const float2& lowerPortion,
												 float occlusion)
{
	// CAS Atomic
	uint64_t assumed, old = *gIllumPortion;
	do
	{
		assumed = old;
		uint64_t avg = AvgIllumPortion(assumed, upperPortion, lowerPortion, occlusion);
		old = atomicCAS(gIllumPortion, assumed, avg);
	} while(assumed != old);
	return old;
}

inline __device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
												 const float4& lowerPortion,
												 float occlusion)
{
	// CAS Atomic
	uint64_t assumed, old = *gIllumPortion;
	do
	{
		assumed = old;
		uint64_t avg = AvgIllumPortion(assumed, lowerPortion, occlusion);
		old = atomicCAS(gIllumPortion, assumed, avg);
	} while(assumed != old);
	return old;
}