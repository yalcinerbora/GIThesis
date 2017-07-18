#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"


__device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
									const float4& upperPortion,
									const float3& lowerPortion);

//__device__ uint64_t AvgAlbedoAndOccupancy(const uint64_t& occupancyPortion,
//										  const float4& albedo,
//										  const float3& normal,
//										  const float occupancy);

__device__ uint64_t AtomicIllumPortionAvg(uint64_t* gIllumPortion,
										  const float4& upperPortion,
										  const float3& lowerPortion);

//__device__ uint64_t AtomicIllumLeafAvg(uint64_t* gIllumUpper,
//									   const float4& albedo,
//									   const float3& normal,
//									   float nodeOccupancy);

//__device__ CSVOIllumination AtomicIllumAvg(CSVOIllumination* gIllum,
//										   const float4& nodeIrradiance,
//										   const float3& nodeNormal,
//										   const float3& nodeLightDir,
//										   const float nodeOccupancy);

inline __device__ uint64_t AvgIllumPortion(const uint64_t& illumPortion,
										   const float4& lowerPortion,
										   const float3& upperPortion)
{
	// Unpack Illumination
	uint2 illumSplit = UnpackWords(illumPortion);
	float4 avgLower = UnpackSVOIrradiance(illumSplit.x);
	float4 avgUpper = UnpackSVONormal(illumSplit.y);
	
	// Divisors
	float counter = avgUpper.w;
	float invCounter = 1.0f / (avgUpper.w + 1.0f);

	// Irradiance Average
	avgLower.x = (counter * avgLower.x + lowerPortion.x) * invCounter;
	avgLower.y = (counter * avgLower.y + lowerPortion.y) * invCounter;
	avgLower.z = (counter * avgLower.z + lowerPortion.z) * invCounter;
	avgLower.w = (counter * avgLower.w + lowerPortion.w) * invCounter;

	// Normal Average
	avgUpper.x = (counter * avgUpper.x + upperPortion.x) * invCounter;
	avgUpper.y = (counter * avgUpper.y + upperPortion.y) * invCounter;
	avgUpper.z = (counter * avgUpper.z + upperPortion.z) * invCounter;

	// Average
	avgUpper.w += 1.0f;

	return PackWords(PackSVONormal(avgUpper), PackSVOIrradiance(avgLower));
}

//inline __device__ uint64_t AvgAlbedoAndOccupancy(const uint64_t& lowerPortion,
//												 const float4& albedo,
//												 const float3& normal,
//												 const float occupancy)
//{
//	// Unpack Illumination
//	uint2 illumSplit = UnpackWords(lowerPortion);
//
//	float avgOccupancy, counter;
//	float4 avgAlbedo = UnpackSVOIrradiance(illumSplit.x);
//	float3 avgNormal = UnpackSVONormalLeaf(avgOccupancy, counter, illumSplit.y);
//	assert(counter < 256.0f);
//
//	// Divisors
//	float invCounter = 1.0f / (counter + 1.0f);
//	
//	// Irradiance Average
//	avgAlbedo.x = (counter * avgAlbedo.x + albedo.x) * invCounter;
//	avgAlbedo.y = (counter * avgAlbedo.y + albedo.y) * invCounter;
//	avgAlbedo.z = (counter * avgAlbedo.z + albedo.z) * invCounter;
//	avgAlbedo.w = (counter * avgAlbedo.w + albedo.w) * invCounter;
//
//	//avgAlbedo.x = fminf(1.0f, avgAlbedo.x);
//	//avgAlbedo.y = fminf(1.0f, avgAlbedo.y);
//	//avgAlbedo.z = fminf(1.0f, avgAlbedo.z);
//	//avgAlbedo.w = fminf(1.0f, avgAlbedo.w);
//
//	// Normal Average
//	avgNormal.x = (counter * avgNormal.x + normal.x) * invCounter;
//	avgNormal.y = (counter * avgNormal.y + normal.y) * invCounter;
//	avgNormal.z = (counter * avgNormal.z + normal.z) * invCounter;
//	
//	//if(avgNormal.z < -1.0f) printf("\f ", avgNormal.z);
//	//if(avgNormal.z > 1.0f) printf("\f ", avgNormal.z);
//	//if(avgNormal.z != avgNormal.z) printf("\f ", avgNormal.z);
//
//	// Occupancy
//	//avgOccupancy += occupancy;
//	//avgOccupancy = fminf(1.0f, avgOccupancy);
//
//	// Average
//	counter += 1.0f;
//
//	return PackWords(PackSVONormalLeaf(avgNormal, avgOccupancy, counter),
//					 PackSVOIrradiance(avgAlbedo));
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

//inline __device__ uint64_t AtomicIllumLeafAvg(uint64_t* gIllumLower,
//											  const float4& albedo,
//											  const float3& normal,
//											  float nodeOccupancy)
//{
//	uint64_t assumed, old = *gIllumLower;
//	do
//	{
//		assumed = old;
//		uint64_t avg = AvgAlbedoAndOccupancy(assumed, albedo, normal, nodeOccupancy);
//		old = atomicCAS(gIllumLower, assumed, avg);
//	} while(assumed != old);
//	return old;
//}

//inline __device__ CSVOIllumination AtomicIllumAvg(CSVOIllumination* gIllum,
//												  const float4& nodeIrradiance,
//												  const float3& nodeNormal,
//												  const float3& nodeLightDir,
//												  const float nodeOccupancy)
//{
//	//// Non-atomic
//	//unsigned int pOccup = PackSVOOccupancy({nodeOccupancy,nodeOccupancy,nodeOccupancy,nodeOccupancy});
//	//unsigned int pLDir = PackSVOLightDir({nodeLightDir.x,nodeLightDir.y,nodeLightDir.z,0});
//	//uint64_t lower = PackWords(pOccup, pLDir);
//
//	//unsigned int pIrrad = PackSVOIrradiance(nodeIrradiance);
//	//unsigned int pNormal = PackSVONormal({nodeNormal.x,nodeNormal.y,nodeNormal.z,0});
//	//uint64_t upper = PackWords(pIrrad, pNormal);
//	//return CSVOIllumination{lower, upper};
//
//	uint64_t lower = AtomicIllumPortionAvg(&(gIllum->irradiancePortion),
//										  nodeIrradiance,
//										  nodeNormal);
//
//	float4 anisoOccupancy = float4{nodeOccupancy, nodeOccupancy, nodeOccupancy, nodeOccupancy};
//	uint64_t upper = AtomicIllumPortionAvg(&(gIllum->occupancyPortion),
//										   anisoOccupancy,
//										   nodeLightDir);
//
//	return CSVOIllumination{lower, upper};
//}