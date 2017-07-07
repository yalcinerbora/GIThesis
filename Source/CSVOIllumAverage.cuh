#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"


__device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
									const float4& upperPortion,
									const float3& lowerPortion);

__device__ uint64_t AvgAlbedoAndOccupancy(const uint64_t& occupancyPortion,
										  const float4& albedo,
										  const float occupancy);

__device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
										  const float4& upperPortion,
										  const float3& lowerPortion);

__device__ uint64_t AtomicIllumLeafAvg(uint64_t* gIllumUpper,
									   const float4& albedo,
									   float nodeOccupancy);

__device__ CSVOIllumination AtomicIllumAvg(CSVOIllumination* gIllum,
										   const float4& nodeIrradiance,
										   const float3& nodeNormal,
										   const float3& nodeLightDir,
										   const float nodeOccupancy);

inline __device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
										   const float4& upperPortion,
										   const float3& lowerPortion)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(illumPortion);
	float4 avgUpper = UnpackSVOIrradiance(illumSplit.x);
	float4 avgLower = UnpackSVONormal(illumSplit.y);
	
	// Divisors
	float invCount = 1.0f / (avgLower.w + 1.0f);

	// Irradiance Average
	avgUpper.x = (avgLower.w * avgUpper.x + upperPortion.x) * invCount;
	avgUpper.y = (avgLower.w * avgUpper.y + upperPortion.y) * invCount;
	avgUpper.z = (avgLower.w * avgUpper.z + upperPortion.z) * invCount;
	avgUpper.w = (avgLower.w * avgUpper.w + upperPortion.w) * invCount;
	
	avgUpper.x = fminf(1.0f, avgUpper.x);
	avgUpper.y = fminf(1.0f, avgUpper.y);
	avgUpper.z = fminf(1.0f, avgUpper.z);
	avgUpper.w = fminf(1.0f, avgUpper.w);

	// Normal Average
	avgLower.x = (avgLower.w * avgLower.x + lowerPortion.x) * invCount;
	avgLower.y = (avgLower.w * avgLower.y + lowerPortion.y) * invCount;
	avgLower.z = (avgLower.w * avgLower.z + lowerPortion.z) * invCount;

	// Increment Counter
	avgLower.w += 1.0f;

	return PackWords(PackSVOIrradiance(avgUpper), 
					 PackSVONormal(avgLower));
}

inline __device__ uint64_t AvgAlbedoAndOccupancy(const uint64_t& occupancyPortion,
												 const float4& albedo,
												 const float occupancy)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(occupancyPortion);
	float4 avgAlbedo = UnpackSVOIrradiance(illumSplit.x);
	float avgOccup = UnpackSVOLeafOccupancy(illumSplit.y);
	
	// Divisors
	float invCount = 1.0f / (avgOccup + occupancy);
	
	// Irradiance Average
	avgAlbedo.x = (avgOccup * avgAlbedo.x + albedo.x) * invCount;
	avgAlbedo.y = (avgOccup * avgAlbedo.y + albedo.y) * invCount;
	avgAlbedo.z = (avgOccup * avgAlbedo.z + albedo.z) * invCount;
	avgAlbedo.w = (avgOccup * avgAlbedo.w + albedo.w) * invCount;

	avgAlbedo.x = fminf(1.0f, avgAlbedo.x);
	avgAlbedo.y = fminf(1.0f, avgAlbedo.y);
	avgAlbedo.z = fminf(1.0f, avgAlbedo.z);
	avgAlbedo.w = fminf(1.0f, avgAlbedo.w);

	// Normal Average
	avgOccup += occupancy;

	return PackWords(PackSVOIrradiance(avgAlbedo),
					 PackSVOLeafOccupancy(avgOccup));
}

inline __device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
												 const float4& upperPortion,
												 const float3& lowerPortion)
{
	// CAS Atomic
	uint64_t assumed, old = *gIllumPortion;
	do
	{
		assumed = old;
		uint64_t avg = AvgIllumPortion(assumed, upperPortion, lowerPortion);
		old = atomicCAS(gIllumPortion, assumed, avg);
	} while(assumed != old);
	return old;
}

inline __device__ uint64_t AtomicIllumLeafAvg(uint64_t* gIllumUpper,
											  const float4& albedo,
											  float nodeOccupancy)
{
	uint64_t assumed, old = *gIllumUpper;
	do
	{
		assumed = old;
		uint64_t avg = AvgAlbedoAndOccupancy(assumed, albedo, nodeOccupancy);
		old = atomicCAS(gIllumUpper, assumed, avg);
	} while(assumed != old);
	return old;
}

inline __device__ CSVOIllumination AtomicIllumAvg(CSVOIllumination* gIllum,
												  const float4& nodeIrradiance,
												  const float3& nodeNormal,
												  const float3& nodeLightDir,
												  const float nodeOccupancy)
{
	//// Non-atomic
	//unsigned int pOccup = PackSVOOccupancy({nodeOccupancy,nodeOccupancy,nodeOccupancy,nodeOccupancy});
	//unsigned int pLDir = PackSVOLightDir({nodeLightDir.x,nodeLightDir.y,nodeLightDir.z,0});
	//uint64_t lower = PackWords(pOccup, pLDir);

	//unsigned int pIrrad = PackSVOIrradiance(nodeIrradiance);
	//unsigned int pNormal = PackSVONormal({nodeNormal.x,nodeNormal.y,nodeNormal.z,0});
	//uint64_t upper = PackWords(pIrrad, pNormal);
	//return CSVOIllumination{lower, upper};

	uint64_t lower = AtomicIllumPortionAvg(&(gIllum->irradiancePortion),
										  nodeIrradiance,
										  nodeNormal);

	float4 anisoOccupancy = float4{nodeOccupancy, nodeOccupancy, nodeOccupancy, nodeOccupancy};
	uint64_t upper = AtomicIllumPortionAvg(&(gIllum->occupancyPortion),
										   anisoOccupancy,
										   nodeLightDir);

	return CSVOIllumination{lower, upper};
}