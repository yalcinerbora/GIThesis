#pragma once

#include "CSVOFunctions.cuh"
#include "CVoxelFunctions.cuh"
#include "CSVOLightInject.cuh"

inline __device__ uint64_t AverageColor(const uint64_t& matPortion,
										const float4& colorUnpack,
										const float2& illumXY,
										float occupancy)
{
	// Unpack Material	
	CSVOColor avgColorPacked = UnpackSVOMaterialColorOrNormal(matPortion);
	CSVOWeight avgWeightPacked = UnpackSVOMaterialWeight(matPortion);

	half weight;
	float4 avgColor = UnpackSVOColor(avgColorPacked);
	float2 avgIllum = UnpackSVOWeight(weight, avgWeightPacked);

	float ratio = __half2float(weight) / (__half2float(weight) + occupancy);
	//float ratio = weight / (weight + occupancy);

	// Color Avg
	avgColor.x = (ratio * avgColor.x) + (colorUnpack.x * occupancy / (__half2float(weight) + occupancy));
	avgColor.y = (ratio * avgColor.y) + (colorUnpack.y * occupancy / (__half2float(weight) + occupancy));
	avgColor.z = (ratio * avgColor.z) + (colorUnpack.z * occupancy / (__half2float(weight) + occupancy));
	avgColor.w = (ratio * avgColor.w) + (colorUnpack.w * occupancy / (__half2float(weight) + occupancy));
	weight = __float2half(__half2float(weight) + occupancy);
	//weight += occupancy;

	// Illum Avg
	// TODO lazy simple overwrite since there is single light
	avgIllum.x = illumXY.x;
	avgIllum.y = illumXY.y;

	avgColorPacked = PackSVOColor(avgColor);
	avgWeightPacked = PackSVOWeight(avgIllum, weight);
	return PackSVOMaterialPortion(avgColorPacked, avgWeightPacked);
}

inline __device__ uint64_t AverageNormal(const uint64_t& matPortion,
										 const float4& normalUnpack,
										 const float2& illumZW,
										 float occupancy)
{
	// Unpack Material	
	CVoxelNorm avgNormalPacked = UnpackSVOMaterialColorOrNormal(matPortion);
	CSVOWeight avgWeightPacked = UnpackSVOMaterialWeight(matPortion);

	half weight;
	float4 avgNormal = UnpackSVONormal(avgNormalPacked);
	float2 avgIllum = UnpackSVOWeight(weight, avgWeightPacked);

	float ratio = __half2float(weight) / (__half2float(weight) + occupancy);
	//float ratio = weight / (weight + occupancy);

	// Normal Avg
	avgNormal.x = (ratio * avgNormal.x) + (normalUnpack.x * occupancy / (__half2float(weight) + occupancy));
	avgNormal.y = (ratio * avgNormal.y) + (normalUnpack.y * occupancy / (__half2float(weight) + occupancy));
	avgNormal.z = (ratio * avgNormal.z) + (normalUnpack.z * occupancy / (__half2float(weight) + occupancy));
	avgNormal.w = fminf((avgNormal.w + occupancy), 1.0f);
	weight = __float2half(__half2float(weight) + occupancy);
	//weight += occupancy;

	// Illum Avg
	// TODO lazy simple overwrite since there is single light
	avgIllum.x = illumZW.x;
	avgIllum.y = illumZW.y;

	avgNormalPacked = PackSVONormal(avgNormal);
	avgWeightPacked = PackSVOWeight(avgIllum, weight);
	return PackSVOMaterialPortion(avgNormalPacked, avgWeightPacked);
}

inline __device__ uint64_t AtomicColorAvg(uint64_t* gMaterialColorPortion,
										  const CSVOColor& color,
										  const float2& illumXY,
										  float occupancy)
{
	float4 colorUnpack = UnpackSVOColor(color);
	uint64_t assumed, old = *gMaterialColorPortion;
	do
	{
		assumed = old;
		old = atomicCAS(gMaterialColorPortion, assumed,
						AverageColor(assumed, colorUnpack, illumXY, occupancy));
	} while(assumed != old);
	return old;
}

inline __device__ uint64_t AtomicNormalAvg(uint64_t* gMaterialNormalPortion,
										   const CVoxelNorm& voxelNormal,
										   const float2& illumZW,
										   float occupancy)
{
	float4 normalUnpack = UnpackSVONormal(voxelNormal);
	uint64_t assumed, old = *gMaterialNormalPortion;

	do
	{
		assumed = old;
		old = atomicCAS(gMaterialNormalPortion, assumed,
						AverageNormal(assumed, normalUnpack, illumZW, occupancy));
	} while(assumed != old);
	return old;
}

inline __device__ CSVOMaterial AtomicAvg(CSVOMaterial* gMaterial,
										 const CSVOColor& color,
										 const CVoxelNorm& voxelNormal,
										 const float4& illumDir,
										 const float& occupancy)
{

	uint64_t avgC = AtomicColorAvg(&(gMaterial->colorPortion), color,
	{illumDir.x, illumDir.y}, occupancy);
	uint64_t avgN = AtomicNormalAvg(&(gMaterial->normalPortion), voxelNormal,
	{illumDir.z, illumDir.w}, occupancy);
	return CSVOMaterial{avgC, avgN};
}