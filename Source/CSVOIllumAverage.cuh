#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"


__device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
									const float4& upperPortion,
									const float3& lowerPortion);

__device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
										  const float4& upperPortion,
										  const float3& lowerPortion);

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

	illumSplit.x = PackSVOIrradiance(avgUpper);
	illumSplit.y = PackSVONormal(avgLower);
	return PackWords(illumSplit.x, illumSplit.y);
}

//inline __device__ uint64_t AvgOccupancyAndLightDir(const uint64_t& occupancyPortion,
//												   const float3& lightDir,
//												   const float occupancy)
//{
//	// Unpack Illumination
//	uint2 illumSplit = UnpackWords(occupancyPortion);
//	float4 avgOccupancy = UnpackSVOOccupancy(illumSplit.x);
//	float4 avgLightDir = UnpackSVOLightDir(illumSplit.y);
//
//	// Divisors
//	float invCount = 1.0f / (avgLightDir.w + 1.0f);
//	
//	// Irradiance Average
//	avgOccupancy.x = (avgLightDir.w * avgOccupancy.x + occupancy) * invCount;
//	avgOccupancy.y = (avgLightDir.w * avgOccupancy.y + occupancy) * invCount;
//	avgOccupancy.z = (avgLightDir.w * avgOccupancy.z + occupancy) * invCount;
//	avgOccupancy.w = (avgLightDir.w * avgOccupancy.w + occupancy) * invCount;
//
//	avgOccupancy.x = fminf(1.0f, avgOccupancy.x);
//	avgOccupancy.y = fminf(1.0f, avgOccupancy.y);
//	avgOccupancy.z = fminf(1.0f, avgOccupancy.z);
//	avgOccupancy.w = fminf(1.0f, avgOccupancy.w);
//
//	// Normal Average
//	avgLightDir.x = (avgLightDir.w * avgLightDir.x + lightDir.x * occupancy) * invCount;
//	avgLightDir.y = (avgLightDir.w * avgLightDir.y + lightDir.y * occupancy) * invCount;
//	avgLightDir.z = (avgLightDir.w * avgLightDir.z + lightDir.z * occupancy) * invCount;
//
//	avgLightDir.w += 1.0f;
//
//	illumSplit.x = PackSVOOccupancy(avgOccupancy);
//	illumSplit.y = PackSVOLightDir(avgLightDir);
//	return PackWords(illumSplit.x, illumSplit.y);
//}


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

//inline __device__ uint64_t AtomicOccupLightDirAvg(uint64_t* gIllumUpper,
//												  const float3& nodeLightDir,
//												  float nodeOccupancy)
//{
//	uint64_t assumed, old = *gIllumUpper;
//	do
//	{
//		assumed = old;
//		old = atomicCAS(gIllumUpper, assumed,
//						AvgOccupancyAndLightDir(assumed, 
//												nodeLightDir,
//												nodeOccupancy));
//	} while(assumed != old);
//	return old;
//}

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