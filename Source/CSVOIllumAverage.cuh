#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"
//#include "CSVOLightInject.cuh"

inline __device__ uint64_t AvgIrradianceAndNormal(const uint64_t& illumPortion,
												  const float4& irradiance,
												  const float3& normal,
												  const float occupancy)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(illumPortion);
	float4 avgIrrad = UnpackSVOIrradiance(illumSplit.x);
	float4 avgNormal = UnpackSVONormal(illumSplit.y);
	
	// Divisors
	float invCount = 1.0f / (avgNormal.w + 1.0f);

	// Irradiance Average
	avgIrrad.x = (avgNormal.w * avgIrrad.x + irradiance.x * occupancy) * invCount;
	avgIrrad.y = (avgNormal.w * avgIrrad.y + irradiance.y * occupancy) * invCount;
	avgIrrad.z = (avgNormal.w * avgIrrad.z + irradiance.z * occupancy) * invCount;
	avgIrrad.w = (avgNormal.w * avgIrrad.w + irradiance.w * occupancy) * invCount;
	
	// Normal Average
	avgNormal.x = (avgNormal.w * avgNormal.x + normal.x * occupancy) * invCount;
	avgNormal.y = (avgNormal.w * avgNormal.y + normal.y * occupancy) * invCount;
	avgNormal.z = (avgNormal.w * avgNormal.z + normal.z * occupancy) * invCount;

	avgNormal.w += 1.0f;

	illumSplit.x = PackSVOIrradiance(avgIrrad);
	illumSplit.y = PackSVONormal(avgNormal);
	return PackWords(illumSplit.x, illumSplit.y);
}

inline __device__ uint64_t AvgOccupancyAndLightDir(const uint64_t& occupancyPortion,
												   const float3& lightDir,
												   const float occupancy)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(occupancyPortion);
	float4 avgOccupancy = UnpackSVOOccupancy(illumSplit.x);
	float4 avgLightDir = UnpackSVOLightDir(illumSplit.y);

	// Divisors
	float invCount = 1.0f / (avgLightDir.w + 1.0f);
	
	// Irradiance Average
	avgOccupancy.x = (avgLightDir.w * avgOccupancy.x + occupancy) * invCount;
	avgOccupancy.y = (avgLightDir.w * avgOccupancy.y + occupancy) * invCount;
	avgOccupancy.z = (avgLightDir.w * avgOccupancy.z + occupancy) * invCount;
	avgOccupancy.w = (avgLightDir.w * avgOccupancy.w + occupancy) * invCount;

	// Normal Average
	avgLightDir.x = (avgLightDir.w * avgLightDir.x + lightDir.x * occupancy) * invCount;
	avgLightDir.y = (avgLightDir.w * avgLightDir.y + lightDir.y * occupancy) * invCount;
	avgLightDir.z = (avgLightDir.w * avgLightDir.z + lightDir.z * occupancy) * invCount;

	avgLightDir.w += 1.0f;

	illumSplit.x = PackSVOOccupancy(avgOccupancy);
	illumSplit.y = PackSVOLightDir(avgLightDir);
	return PackWords(illumSplit.x, illumSplit.y);
}


inline __device__ uint64_t AtomicIllumNormalAvg(uint64_t* gIllumLower,
												const float4& nodeIrradiance,
												const float3& nodeNormal,
												float nodeOccupancy)
{
	// CAS Atomic
	uint64_t assumed, old = *gIllumLower;
	do
	{
		assumed = old;
		old = atomicCAS(gIllumLower, assumed,
						AvgIrradianceAndNormal(assumed, 
											   nodeIrradiance, 
											   nodeNormal, 
											   nodeOccupancy));
	} while(assumed != old);
	return old;
}

inline __device__ uint64_t AtomicOccupLightDirAvg(uint64_t* gIllumUpper,
												  const float3& nodeLightDir,
												  float nodeOccupancy)
{
	uint64_t assumed, old = *gIllumUpper;
	do
	{
		assumed = old;
		old = atomicCAS(gIllumUpper, assumed,
						AvgOccupancyAndLightDir(assumed, 
												nodeLightDir,
												nodeOccupancy));
	} while(assumed != old);
	return old;
}

inline __device__ CSVOIllumination AtomicIllumAvg(CSVOIllumination* gIllum,
												  const float4& nodeIrradiance,
												  const float3& nodeNormal,
												  const float3& nodeLightDir,
												  const float nodeOccupancy)
{

	uint64_t lower = AtomicIllumNormalAvg(&(gIllum->irradiancePortion),
										  nodeIrradiance,
										  nodeNormal,
										  nodeOccupancy);
	uint64_t upper = AtomicOccupLightDirAvg(&(gIllum->occupancyPortion),
											nodeLightDir,
											nodeOccupancy);

	return CSVOIllumination{lower, upper};
}