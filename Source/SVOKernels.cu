#include "SVOKernels.cuh"
#include "GISparseVoxelOctree.h"
#include "GIVoxelPages.h"
#include "CSVOHash.cuh"
#include "CVoxelFunctions.cuh"
#include "CSVOLightInject.cuh"
#include "CSVOIllumAverage.cuh"
#include "CSVONodeAlloc.cuh"
#include <cuda.h>

//
//
//
//	unsigned int location;
//	CSVONode* node = nullptr;
//	for(unsigned int i = octreeParams.DenseLevel; i <= level; i++)
//	{
//		CSVONode* node = nullptr;
//		if(i == svoConstants.denseDepth)
//		{
//			uint3 levelVoxId = CalculateLevelVoxId(voxelPos, i, svoConstants.totalDepth);
//			node = gSVODense +
//				svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
//				svoConstants.denseDim * levelVoxId.y +
//				levelVoxId.x;
//		}
//		else
//		{
//			node = gSVOSparse + gLevelOffsets[levelIndex] + location;
//		}
//
//		// Allocate (or acquire) next location
//		location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1]);
//		assert(location < gLevelTotalSizes[levelIndex + 1]);
//
//		// Offset child
//		unsigned int childId = CalculateLevelChildId(currentVoxPos, i + 1, svoConstants.totalDepth);
//		location += childId;
//	}
//}
//
////inline __device__ unsigned int FindDenseChildren(const uint3& parentIndex,
////                                                 const unsigned int childId,
////                                                 const unsigned int levelDim)
////{
////    // Go down 1 lvl
////    uint3 childIndex = parentIndex;
////    childIndex.x *= 2;
////    childIndex.y *= 2;
////    childIndex.z *= 2;
////
////    uint3 offsetIndex =
////    {
////        childId % 2,
////        childId / 2,
////        childId / 4
////    };
////    childIndex.x += offsetIndex.x;
////    childIndex.y += offsetIndex.y;
////    childIndex.z += offsetIndex.z;
////
////    unsigned int childLvlDim = levelDim << 1;
////    unsigned int linearChildId = childIndex.z * childLvlDim * childLvlDim +
////        childIndex.y * childLvlDim +
////        childIndex.z;
////    return linearChildId;
////}
////
//
//__global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,
//										  cudaSurfaceObject_t sDenseMat,
//
//										  const CSVONode* gSVODense,
//										  const CSVONode* gSVOSparse,
//
//										  const unsigned int* gLevelOffsets,
//										  const unsigned int& gSVOLevelOffset,
//										  const unsigned int& gSVONextLevelOffset,
//
//										  const unsigned int levelNodeCount,
//										  const unsigned int matOffset,
//										  const unsigned int currentLevel,
//										  const CSVOConstants& svoConstants)
//{
//	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int nodeId = globalId / 2;
//
//	// Cull if out of range
//	if(nodeId > levelNodeCount) return;
//
//	// Read Sibling Materials
//	const CSVONode* n = (currentLevel == svoConstants.denseDepth) ? gSVODense : gSVOSparse;
//	CSVONode node = n[gSVOLevelOffset + nodeId];
//
//	// Cull if there is no node no need to average
//	if(node == 0xFFFFFFFF) return;
//
//	// Only fetch parent when there a potential to have one
//	bool fetchParentMat = ((svoConstants.totalDepth - currentLevel) < svoConstants.numCascades);
//
//	uint64_t parentMat;
//	if(globalId % 2 == 0) parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + nodeId].colorPortion : 0x0;
//	else parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + nodeId].normalPortion : 0x0;
//
//	// Average Portion
//	// Material Data
//	unsigned int count = 0;
//	float4 avgSegment1 = {0.0f, 0.0f, 0.0f, 0.0f};
//	float4 avgSegment2 = {0.0f, 0.0f, 0.0f, 0.0f};
//
//	// Parent Incorporate
//	if(parentMat != 0x0)
//	{
//		if(globalId % 2 == 0)
//		{
//			CSVOColor colorPacked = UnpackSVOMaterialColorOrNormal(parentMat);
//			float4 color = UnpackSVOColor(colorPacked);
//		
//			avgSegment1.x = 8 * color.x;
//			avgSegment1.y = 8 * color.y;
//			avgSegment1.z = 8 * color.z;
//			avgSegment1.w = 8 * color.w;
//		}
//		else
//		{
//			CVoxelNorm normalPacked = UnpackSVOMaterialColorOrNormal(parentMat);
//			float4 normal = UnpackSVONormal(normalPacked);
//
//			avgSegment2.x = 8 * normal.x;
//			avgSegment2.y = 8 * normal.y;
//			avgSegment2.z = 8 * normal.z;
//			avgSegment2.w = 8 * normal.w;
//		}
//		count += 8;
//	}
//
//	// Average
//	if(node != 0xFFFFFFFF)
//	{
//		#pragma unroll
//		for(unsigned int i = 0; i < 8; i++)
//		{
//			unsigned int currentNodeId = node + i;
//			if(globalId % 2 == 0)
//			{
//				uint64_t mat = gSVOMat[matOffset + gSVONextLevelOffset + currentNodeId].colorPortion;
//				if(mat == 0x0) continue;
//
//				CSVOColor colorPacked = UnpackSVOMaterialColorOrNormal(mat);
//				float4 color = UnpackSVOColor(colorPacked);
//
//				avgSegment1.x += color.x;
//				avgSegment1.y += color.y;
//				avgSegment1.z += color.z;
//				avgSegment1.w += color.w;
//			}
//			else
//			{
//				uint64_t mat = gSVOMat[matOffset + gSVONextLevelOffset + currentNodeId].normalPortion;
//				if(mat == 0x0) continue;
//
//				CVoxelNorm normalPacked = UnpackSVOMaterialColorOrNormal(mat);
//				float4 normal = UnpackSVONormal(normalPacked);
//
//				avgSegment2.x += normal.x;
//				avgSegment2.y += normal.y;
//				avgSegment2.z += normal.z;
//				avgSegment2.w += normal.w;
//			}
//			count++;
//		}
//	}
//
//	// Divide by Count
//	if(count == 0) count = 1.0f;
//	float countInv = 1.0f / static_cast<float>(count);
//	avgSegment1.x *= countInv;
//	avgSegment1.y *= countInv;
//	avgSegment1.z *= countInv;
//	avgSegment1.w *= countInv;
//
//	avgSegment2.x *= countInv;
//	avgSegment2.y *= countInv;
//	avgSegment2.z *= countInv;
//	avgSegment2.w *= (count > 8) ? 0.0625f : 0.125f;
//
//	// Pack and Store	
//	uint64_t averageValue;
//	if(globalId % 2 == 0)
//	{
//		CSVOColor colorPacked = PackSVOColor(avgSegment1);		
//		averageValue = PackSVOMaterialPortion(colorPacked, 0x0);
//	}
//	else
//	{		
//		CVoxelNorm normPacked = PackSVONormal(avgSegment2);
//		averageValue = PackSVOMaterialPortion(normPacked, 0x0);
//	}
//	
//    if(currentLevel == svoConstants.denseDepth)
//    {
//        int3 dim =
//        {
//            static_cast<int>(nodeId % svoConstants.denseDim),
//            static_cast<int>((nodeId / svoConstants.denseDim) % svoConstants.denseDim),
//            static_cast<int>(nodeId / (svoConstants.denseDim * svoConstants.denseDim))
//        };
//        uint2 data =
//        {
//            static_cast<unsigned int>(averageValue & 0x00000000FFFFFFFF),
//            static_cast<unsigned int>(averageValue >> 32)
//        };
//		int dimX = (globalId % 2 == 0) ? (dim.x * sizeof(uint4)) : (dim.x * sizeof(uint4) + sizeof(uint2));
//        surf3Dwrite(data, sDenseMat, dimX, dim.y, dim.z);
//    }
//    else
//    {
//		if(globalId % 2 == 0) gSVOMat[matOffset + gSVOLevelOffset + nodeId].colorPortion = averageValue;
//		else gSVOMat[matOffset + gSVOLevelOffset + nodeId].normalPortion = averageValue;
//    }
//}
//
//__global__ void SVOReconstructAverageNode(cudaSurfaceObject_t sDenseMatChild,
//                                          cudaSurfaceObject_t sDenseMatParent,
//
//                                          const unsigned int parentSize)
//{
//    // Linear Id
//    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//    unsigned int parentId = globalId / GI_DENSE_WORKER_PER_PARENT;
//
//    // 3D Id
//    char3 idMap = voxLookup[globalId % GI_DENSE_WORKER_PER_PARENT];
//    uint3 parentId3D =
//    {
//        static_cast<unsigned int>(parentId % parentSize),
//        static_cast<unsigned int>((parentId / parentSize) % parentSize),
//        static_cast<unsigned int>(parentId / (parentSize * parentSize))
//    };
//    uint3 childId3D =
//    {
//        parentId3D.x * 2 + idMap.x,
//        parentId3D.y * 2 + idMap.y,
//        parentId3D.z * 2 + idMap.z
//    };
//
//    // 3D Fetch
//    uint4 data;
//    surf3Dread(&data, sDenseMatChild,
//               childId3D.x * sizeof(uint4),
//               childId3D.y,
//               childId3D.z);
//
//    // Data
//    unsigned int count = (data.x == 0 && 
//						  data.y == 0 && 
//						  data.z == 0 && 
//						  data.w == 0) ? 0 : 1;
//    float4 color = UnpackSVOColor(data.x);
//    float4 normal = UnpackSVONormal(data.w);
//
//    // Average	
//    #pragma unroll
//    for(int offset = GI_DENSE_WORKER_PER_PARENT / 2; offset > 0; offset /= 2)
//    {
//        color.x += __shfl_down(color.x, offset, GI_DENSE_WORKER_PER_PARENT);
//        color.y += __shfl_down(color.y, offset, GI_DENSE_WORKER_PER_PARENT);
//        color.z += __shfl_down(color.z, offset, GI_DENSE_WORKER_PER_PARENT);
//        color.w += __shfl_down(color.w, offset, GI_DENSE_WORKER_PER_PARENT);
//
//        normal.x += __shfl_down(normal.x, offset, GI_DENSE_WORKER_PER_PARENT);
//        normal.y += __shfl_down(normal.y, offset, GI_DENSE_WORKER_PER_PARENT);
//        normal.z += __shfl_down(normal.z, offset, GI_DENSE_WORKER_PER_PARENT);
//        normal.w += __shfl_down(normal.w, offset, GI_DENSE_WORKER_PER_PARENT);
//
//        count += __shfl_down(count, offset, GI_DENSE_WORKER_PER_PARENT);
//    }
//
//    // Division
//    float countInv = 1.0f / ((count != 0) ? float(count) : 1.0f);
//    color.x *= countInv;
//    color.y *= countInv;
//    color.z *= countInv;
//    color.w *= countInv;
//
//    normal.x *= countInv;
//    normal.y *= countInv;
//    normal.z *= countInv;
//	normal.w *= 0.125f;
//
//    data.x = PackSVOColor(color);
//    data.w = PackSVONormal(normal);
//
//    if(globalId % GI_DENSE_WORKER_PER_PARENT == 0 && count != 0)
//    {
//        surf3Dwrite(data, sDenseMatParent,
//                    parentId3D.x * sizeof(uint4),
//                    parentId3D.y,
//                    parentId3D.z);
//    }
//}
//


//inline __device__ unsigned int& Index(uint4& low, 
//									  uint4& high,
//									  const int& index)
//{
//	uint4& loc = (index < 4) ? low : high;
//	return reinterpret_cast<unsigned int*>(&loc)[index % 4];
//}

__global__ void AverageLevelDense(// SVO
								  const CSVOLevel& gCurrentLevel,
								  const CSVOLevelConst& gNextLevel,
								  // Limits
								  const OctreeParameters octreeParams,
								  const uint32_t currentLevelLength)
{
	// Two thread per node
	const unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int nodeId = globalId / 2;
	const unsigned int nodeLocalId = globalId % 2;
	const unsigned int nodeCount = currentLevelLength * currentLevelLength * currentLevelLength;

	// Cull unnecesary threads
	if(nodeId >= nodeCount) return;

	// 3D Node Id
	short3 nodeId3D;
	nodeId3D.x = (nodeId) % currentLevelLength;
	nodeId3D.y = (nodeId / currentLevelLength) % currentLevelLength;
	nodeId3D.z = (nodeId / (currentLevelLength * currentLevelLength));
	nodeId3D.x <<= 1;
	nodeId3D.y <<= 1;
	nodeId3D.z <<= 1;

	// Average Register
	float4 avgLower = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 avgUpper = {0.0f, 0.0f, 0.0f, 0.0f};
	int count = 0;

	#pragma unroll
	for(int i = 0; i < 8; i++)
	{
		short3 childId;
		childId.x = nodeId3D.x + ((i >> 0) & 0x1);
		childId.y = nodeId3D.y + ((i >> 1) & 0x1);
		childId.z = nodeId3D.z + ((i >> 2) & 0x1);
		uint32_t linearId = DenseIndex(childId, currentLevelLength << 1);

		uint64_t childPart = reinterpret_cast<const uint64_t*>(gNextLevel.gLevelIllum + linearId)[nodeLocalId];
		if(childPart != 0x0)
		{
			float4 lowerWord = UnpackSVOIrradiance(UnpackLowerWord(childPart));
			float3 upperWord = ExpandVoxNormal(UnpackUpperWord(childPart));

			avgLower.x += lowerWord.x;
			avgLower.y += lowerWord.y;
			avgLower.z += lowerWord.z;
			avgLower.w += lowerWord.w;

			avgUpper.x += upperWord.x;
			avgUpper.y += upperWord.y;
			avgUpper.z += upperWord.z;
			count++;
		}
	}

	if(count != 0)
	{
		// Division
		float countInv = 1.0f / static_cast<float>(count);
		float lowerDivider = (nodeLocalId == 0) ? countInv : 0.125f;
		avgLower.x *= lowerDivider;
		avgLower.y *= lowerDivider;
		avgLower.z *= lowerDivider;
		avgLower.w *= lowerDivider;

		avgUpper.x *= countInv;
		avgUpper.y *= countInv;
		avgUpper.z *= countInv;
		avgUpper.w = 1.0f;

		// Write averaged value
		uint64_t illumPart = PackWords(PackSVONormal(avgUpper), PackSVOIrradiance(avgLower));
		reinterpret_cast<uint64_t*>(gCurrentLevel.gLevelIllum + nodeId)[nodeLocalId] = illumPart;
	}
}

__global__ void AverageLevelSparse(// SVO
								   const CSVOLevel& gCurrentLevel,
								   const CSVOLevelConst& gNextLevel,
								   // Limits
								   const OctreeParameters octreeParams,
								   const uint32_t nodeCount,
								   const bool isCascadeLevel)
{
	// Two thread per node
	const unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int nodeId = globalId / 2;
	const unsigned int nodeLocalId = globalId % 2;
	
	// Cull unnecesary threads
	if(nodeId >= nodeCount) return;

	//const CSVOLevel& level = gSVOLevels[currentLevel];
	//const CSVOLevel& nextLevel = gSVOLevels[currentLevel + 1];
	const uint32_t nodeChildrenBase = gCurrentLevel.gLevelNodes[nodeId].next;

	// Each Thread will average (and read/write)
	// single double word part of the illumination
	// Average Children
	if(nodeChildrenBase != 0xFFFFFFFF)
	{
		uint64_t illumPart = 0x0;

		// Read potential parent value
		if(isCascadeLevel)
			illumPart = reinterpret_cast<uint64_t*>(gCurrentLevel.gLevelIllum + nodeId)[nodeLocalId];

		float4 avgLower = UnpackSVOIrradiance(UnpackLowerWord(illumPart));
		float4 avgUpper = UnpackSVONormal(UnpackUpperWord(illumPart));
		int count = (illumPart == 0x0) ? 0 : 8;
		float denseDivider = (illumPart == 0x0) ? 0.125f : 0.0625f;

		#pragma unroll
		for(int i = 0; i < 8; i++)
		{
			uint64_t childPart = reinterpret_cast<const uint64_t*>(gNextLevel.gLevelIllum + nodeChildrenBase + i)[nodeLocalId];
			if(childPart != 0x0)
			{
				float4 lowerWord = UnpackSVOIrradiance(UnpackLowerWord(childPart));
				float3 upperWord = ExpandVoxNormal(UnpackUpperWord(childPart));

				avgLower.x += lowerWord.x;
				avgLower.y += lowerWord.y;
				avgLower.z += lowerWord.z;
				avgLower.w += lowerWord.w;

				avgUpper.x += upperWord.x;
				avgUpper.y += upperWord.y;
				avgUpper.z += upperWord.z;

				count++;
			}
		}

		// Division
		float countInv = 1.0f / static_cast<float>(count);
		float lowerDivider = (nodeLocalId == 0) ? countInv : denseDivider;
		avgLower.x *= lowerDivider;
		avgLower.y *= lowerDivider;
		avgLower.z *= lowerDivider;
		avgLower.w *= lowerDivider;

		avgUpper.x *= countInv;
		avgUpper.y *= countInv;
		avgUpper.z *= countInv;
		avgUpper.w = 1.0f;

		// Write averaged value
		illumPart = PackWords(PackSVONormal(avgUpper), PackSVOIrradiance(avgLower));
		reinterpret_cast<uint64_t*>(gCurrentLevel.gLevelIllum + nodeId)[nodeLocalId] = illumPart;
	}
}

__global__ void GenNeigbourPtrs(// SVO
								const CSVOLevel* gSVOLevels,
								uint32_t* gLevelAllocators,
								const uint32_t* gLevelCapacities,
								// Limits
								const OctreeParameters octreeParams,
								const uint32_t nodeCount,
								const uint32_t level)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= nodeCount) return;

	const CSVOLevel& currentLevel = gSVOLevels[level];
	uint32_t voxPosPacked = reinterpret_cast<uint32_t*>(currentLevel.gLevelIllum + globalId * 8)[3];
	if(voxPosPacked != 0x0)
	{
		uint32_t cascadeId = octreeParams.MaxSVOLevel - level;
		const int3 nodePos = ExpandToSVODepth(ExpandVoxPos(voxPosPacked),
											  cascadeId,
											  octreeParams.CascadeCount,
											  octreeParams.CascadeBaseLevel);

		NodeReconstruct(gLevelAllocators,
						gLevelCapacities,
						gSVOLevels,
						nodePos,
						octreeParams,
						level);
	}
}

__global__ void LightInject(const CSVOLevel& gSVOLevel,
							// CascadeRelated
							const CVoxelGrid& gGridInfo,
							// Light Injection Related
							const CLightInjectParameters liParams,
							// Limits
							const OctreeParameters octreeParams,
							const uint32_t nodeCount)
{
	__shared__ float3 sEdgePos;
	__shared__ float sSpan;
	
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Load Required Data
	if(threadIdx.x < 3)
	{
		reinterpret_cast<float*>(&sEdgePos)[threadIdx.x] = 
		reinterpret_cast<const float*>(&gGridInfo.position)[threadIdx.x];
	}
	else if(threadIdx.x < 4)
	{
		sSpan = gGridInfo.span;
	}
	__syncthreads();

	// Loaded to Memory, now can cull unused threads
	if(globalId >= nodeCount) return;

	// Load illum data and world pos
	uint64_t illumPortion = reinterpret_cast<uint64_t*>(gSVOLevel.gLevelIllum + globalId)[0];
	float occupancy = reinterpret_cast<float*>(gSVOLevel.gLevelIllum + globalId)[2];
	uint32_t voxPosPacked = reinterpret_cast<uint32_t*>(gSVOLevel.gLevelIllum + globalId)[3];

	if(illumPortion != 0x0)
	{
		// Unpack Illumination
		uint2 illumSplit = UnpackWords(illumPortion);
		float4 albedo = UnpackSVOIrradiance(illumSplit.x);
		float4 normal = UnpackSVONormal(illumSplit.y);
		occupancy /= normal.w;
		
		//float3 normal3D = Normalize(float3{normal.x, normal.y, normal.z});
		//albedo.x = (normal3D.x + 1.0f) * 0.5f;
		//albedo.y = (normal3D.y + 1.0f) * 0.5f;
		//albedo.z = (normal3D.z + 1.0f) * 0.5f;

		float4 occupF4 = make_float4(occupancy, occupancy, occupancy, occupancy);
		uint32_t* illumlocation = reinterpret_cast<uint32_t*>(gSVOLevel.gLevelIllum + globalId);

		//float3 n = float3{normal.x, normal.y, normal.z};
		//float length = Length(n);
		//illumlocation[0] = PackSVOIrradiance(float4{length, length, length});

		//if(length < 0.01f)
		//{
		//	illumlocation[0] = 0xFF000000FF;
		//}
		////	//illumlocation[0] = PackSVOIrradiance(float4{length, length, length});
		////	//illumlocation[0] = PackSVOIrradiance(albedo);
		////}
		////else
		////{
		////	illumlocation[0] = 0x0;
		////}

		illumlocation[2] = PackSVOOccupancy(occupF4);
	}


	//--------------------------------------------------------------------------
	//	float3 lightDir = {0.0f, 0.0f, 0.0f};
	//	occupancy /= normal.w;

	//	// Do light injection
	//	if(liParams.injectOn)
	//	{
	//		// World Space Position Reconstruction
	//		const int3 voxPos = ExpandVoxPos(voxPosPacked);
	//		float3 worldPos;
	//		worldPos.x = sEdgePos.x + static_cast<float>(voxPos.x) * sSpan;
	//		worldPos.y = sEdgePos.y + static_cast<float>(voxPos.y) * sSpan;
	//		worldPos.z = sEdgePos.z + static_cast<float>(voxPos.z) * sSpan;

	//		//float3 normal3D = float3{normal.x, normal.y, normal.z};
	//		float3 normal3D = Normalize(float3{normal.x, normal.y, normal.z});

	//		//if(normal3D.z != normal3D.z ||
	//		//   normal3D.z != normal3D.z ||
	//		//   normal3D.z != normal3D.z)
	//		//{
	//		//	printf("{%f, %f, %f}", normal3D.z, normal3D.y, normal3D.x);
	//		//}

	//		// Generated Irradiance
	//		float3 irradianceDiffuse = TotalIrradiance(lightDir,
	//												   // Node Params
	//												   worldPos,
	//												   normal3D,
	//												   albedo,
	//												   // Light Parameters
	//												   liParams);

	//		//const CLight& light = liParams.gLightStruct[0];
	//		//float3 worldLight;
	//		//worldLight.x = -light.direction.x;
	//		//worldLight.y = -light.direction.y;
	//		//worldLight.z = -light.direction.z;
	//		//worldLight = Normalize(worldLight);

	//		//// Early bail if back surface
	//		//float NdL = fmaxf(Dot(normal3D, worldLight), 0.0f);
	//		//float3 irradianceDiffuse = {NdL, NdL, NdL};

	//		albedo.x = irradianceDiffuse.x;
	//		albedo.y = irradianceDiffuse.y;
	//		albedo.z = irradianceDiffuse.z;
	//	}
	//	normal.w = 1.0f;

	//	// Occupancy Portion
	//	float4 occupF4 = make_float4(occupancy, occupancy, occupancy, occupancy);
	//	float4 lightDirF4 = make_float4(lightDir.x, lightDir.y, lightDir.z, 1.0f);


	//	//uint32_t occup = PackSVOOccupancy(occupF4);
	//	//printf("writing occupancy %#010X {%f, %f}\n", occup, occupF4.x, occupancy);

	//	// Write illumination
	//	// Naive store failed miserably writing word by word gives better perf
	//	// This write is better perfwise
	//	uint32_t* illumlocation = reinterpret_cast<uint32_t*>(gSVOLevel.gLevelIllum + globalId);
	//	illumlocation[0] = PackSVOIrradiance(albedo);
	//	illumlocation[1] = PackSVONormal(normal);
	//	illumlocation[2] = PackSVOOccupancy(occupF4);
	//	illumlocation[3] = PackSVOLightDir(lightDirF4);
	//}	
}

__global__ void SVOReconstruct(// SVO
							   uint32_t* gLocations,
							   const CSVOLevel* gSVOLevels,
							   uint32_t* gLevelAllocators,
							   const uint32_t* gLevelCapacities,
							   // Voxel Pages
							   const CVoxelPageConst* gVoxelPages,
							   // Limits			
							   const OctreeParameters octreeParams,
							   const uint32_t totalCount)
{
	// Shared Memory for generic data
//	__shared__ CSegmentInfo sSegInfo;
//	__shared__ CMeshVoxelInfo sMeshVoxelInfo;

	__shared__ float sSpan;
	__shared__ float3 sEdgePos;

	// Local Ids
	unsigned int blockLocalId = threadIdx.x;
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int nodeId = globalId;
	unsigned int pageId = nodeId / GIVoxelPages::PageSize;
	unsigned int pageLocalId = nodeId % GIVoxelPages::PageSize;
	unsigned int pageLocalSegmentId = pageLocalId / GIVoxelPages::SegmentSize;
	unsigned int segmentLocalVoxId = pageLocalId % GIVoxelPages::SegmentSize;

	// Get Segments Obj Information Struct
//	CObjectType objType;
	__shared__ CSegmentOccupation sOccupation;
	__shared__ uint8_t sCascadeId;
	bool firstOccurance;
	if(blockLocalId == 0)
	{
		// Load to smem
		// Todo split this into the threadss
		uint16_t segInfo = gVoxelPages[pageId].dSegmentInfo[pageLocalSegmentId].packed;
		sCascadeId = ExpandOnlyCascadeNo(segInfo);
		sOccupation = ExpandOnlyOccupation(segInfo);
		//ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
//		sEdgePos = gGridInfos[cascadeId].position;
//		sSpan = gGridInfos[cascadeId].span;
	}
	__syncthreads();
	//if(blockLocalId != 0)
	//{
	//	ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	//}
	// Full Block Cull
	if(sOccupation == CSegmentOccupation::EMPTY) return;
	assert(sOccupation != CSegmentOccupation::MARKED_FOR_CLEAR);
//	if(blockLocalId == 0)
//	{
////		sMeshVoxelInfo = gBatchVoxelCache[cascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId];
//	}
//	__syncthreads();

	// Now we can cull invalid nodes
	const CVoxelNorm voxelNormPacked = gVoxelPages[pageId].dGridVoxNorm[pageLocalId];
	if(voxelNormPacked == 0xFFFFFFFF) return;

	// Unpack Vox Pos
	const CVoxelPos voxelPosPacked = gVoxelPages[pageId].dGridVoxPos[pageLocalId];

	// Unpack Occupancy
	const CVoxelOccupancy voxOccupPacked = gVoxelPages[pageId].dGridVoxOccupancy[pageLocalId];
	
	// Get Albedo (it is dependant)
	// Find your opengl data and voxel cache
//	const uint16_t& batchId = sSegInfo.batchId;
//	const BatchVoxelCache& batchCache = gBatchVoxelCache[cascadeId * batchCount + batchId];
	// Voxel Ids
//	const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
//	const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelInfo.voxOffset;
	// Voxel Albedo
//	const VoxelAlbedo albedoPacked = batchCache.dVoxelAlbedo[batchLocalVoxelId];


	//// World Space Position Reconstruction
	//float3 occup = ExpandOccupancy(voxOccupPacked);
	//const int3 voxPos = ExpandVoxPos(voxelPosPacked);
	//float3 worldPos;
	//worldPos.x = sEdgePos.x + (static_cast<float>(voxPos.x) + occup.x) * sSpan;
	//worldPos.y = sEdgePos.y + (static_cast<float>(voxPos.y) + occup.y) * sSpan;
	//worldPos.z = sEdgePos.z + (static_cast<float>(voxPos.z) + occup.z) * sSpan;

	////Generated Irradiance
	//float3 lightDir;
	//float3 unpackNormal = ExpandVoxNormal(voxelNormPacked);
	//float4 unpackAlbedo = UnpackSVOIrradiance(albedoPacked);
	////float3 lightDir;
	//float3 irradianceDiffuse = TotalIrradiance(lightDir,
	//										   // Node Params
	//										   worldPos,
	//										   unpackNormal,
	//										   unpackAlbedo,
	//										   // Light Parameters
	//										   liParams);


	// Now we will start allocating all nodes
	// Each node will generate multiple neigbouring nodes (8-neigbour filtering)
	// Two threads are responsible for single node, they store the alocated positions
	// Then each thread will average upper or lower half of the node
	#pragma unroll
	for(int a = 0; a < 8; a++)
	{
		// Convert Linear Loop to 3D
		char3 ijk;
		ijk.x = (a >> 0) & 0x1;
		ijk.y = (a >> 1) & 0x1;
		ijk.z = (a >> 2) & 0x1;
		
		int3 myMetaNeigbour = ExpandVoxPos(voxelPosPacked);
		myMetaNeigbour.x += ijk.x;
		myMetaNeigbour.y += ijk.y;
		myMetaNeigbour.z += ijk.z;

		// Boundary Check
		bool validNode = (myMetaNeigbour.x < octreeParams.CascadeBaseLevelSize &&
						  myMetaNeigbour.y < octreeParams.CascadeBaseLevelSize &&
						  myMetaNeigbour.z < octreeParams.CascadeBaseLevelSize);

		int3 nodePos = ExpandToSVODepth(myMetaNeigbour,
										sCascadeId,
										octreeParams.CascadeCount,
										octreeParams.CascadeBaseLevel);

		// Current level & Parent level
		uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - sCascadeId;
		uint32_t parentLevel = cascadeMaxLevel - 1;

		// Allocate and Average Illumination Values
		uint32_t nodeLocation = 0xFFFFFFFF;
		if(validNode)
		{
			nodeLocation = TraverseAndAllocate(// SVO
											   gLevelAllocators,
											   gLevelCapacities,
											   gSVOLevels,
											   // Node Related
											   nodePos,
											   // Constants
											   octreeParams,
											   cascadeMaxLevel);
		}
		gLocations[a * totalCount + nodeId] = nodeLocation;

			//// Parent Vox Pos Calculation
			//int3 parentPos = nodePos;
			//parentPos.x >>= 1;
			//parentPos.y >>= 1;
			//parentPos.z >>= 1;
			//CVoxelPos parentPosPacked = PackNodeId(parentPos, parentLevel, octreeParams.CascadeCount,
			//									   octreeParams.CascadeBaseLevel, octreeParams.MaxSVOLevel);

			// TODO: write this

			// Add these Node Locations

//			// Calculte this nodes occupancy
//			float3 weights = ExpandOccupancy(voxOccupPacked);
//			float3 volume;
//			volume.x = (ijk.x == 1) ? weights.x : (1.0f - weights.x);
//			volume.y = (ijk.y == 1) ? weights.y : (1.0f - weights.y);
//			volume.z = (ijk.z == 1) ? weights.z : (1.0f - weights.z);
//			float occupancy = volume.x * volume.y * volume.z;
////			float occupancy = 1.0f;
//
//			// Unpack albedo and voxel normal for average
//			float4 unpackAlbedo = UnpackSVOIrradiance(albedoPacked);
//			unpackAlbedo.x = irradianceDiffuse.x;
//			unpackAlbedo.y = irradianceDiffuse.y;
//			unpackAlbedo.z = irradianceDiffuse.z;
//
//			float3 unpackNormal = ExpandVoxNormal(voxelNormPacked);
//
//			// Atomic Average to Location
//			CSVOIllumination* illumNode = gSVOLevels[cascadeMaxLevel].gLevelIllum + nodeLocation;
//			AtomicIllumPortionAvg(reinterpret_cast<uint64_t*>(illumNode),
//								  unpackAlbedo, unpackNormal);
//			AtomicIllumPortionAvg(reinterpret_cast<uint64_t*>(illumNode + 1),
//								  unpackAlbedo, unpackNormal);
			//atomicAdd(reinterpret_cast<float*>(illumNode) + 2, occupancy);

			//// Write Locations
			//uint32_t packedVoxPos = PackVoxPos(myMetaNeigbour);
			//reinterpret_cast<uint32_t*>(illumNode)[3] = packedVoxPos | 0x80000000;
			//if(nodeLocation != (nodeLocation / 8 * 8))
			//illumNode = gSVOLevels[cascadeMaxLevel].gLevelIllum + (nodeLocation / 8 * 8);
			//reinterpret_cast<uint32_t*>(illumNode)[3] = packedVoxPos | 0x80000000;
	}
}

__global__ void SVOLightInject(// SVO
							   const uint32_t* gLocations,
							   const CSVOLevel* gSVOLevels,
							   // Voxel Pages
							   const CVoxelPageConst* gVoxelPages,
							   // Cache Data (for Voxel Albedo)
							   const BatchVoxelCache* gBatchVoxelCache,
							   // Inject Related
							   const CVoxelGrid* gGridInfos,
							   const CLightInjectParameters liParams,
							   // Limits			
							   const OctreeParameters octreeParams,
							   const uint32_t batchCount,
							   const uint32_t totalCount
)
{
	// Shared Memory for generic data
	__shared__ CSegmentInfo sSegInfo;
	__shared__ CSegmentOccupation sOccupation;
	__shared__ uint8_t sCascadeId;
	__shared__ uint32_t sMeshVoxelOffset;

	__shared__ float sSpan;
	__shared__ float3 sEdgePos;

	// Local Ids
	unsigned int blockLocalId = threadIdx.x;
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int nodeId = globalId;
	unsigned int pageId = nodeId / GIVoxelPages::PageSize;
	unsigned int pageLocalId = nodeId % GIVoxelPages::PageSize;
	unsigned int pageLocalSegmentId = pageLocalId / GIVoxelPages::SegmentSize;
	unsigned int segmentLocalVoxId = pageLocalId % GIVoxelPages::SegmentSize;

	if(blockLocalId == 0)
	{
		// Load to smem
		// Todo split this into the threadss
		sSegInfo = gVoxelPages[pageId].dSegmentInfo[pageLocalSegmentId];
		sCascadeId = ExpandOnlyCascadeNo(sSegInfo.packed);
		sOccupation = ExpandOnlyOccupation(sSegInfo.packed);
		sMeshVoxelOffset = gBatchVoxelCache[sCascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId].voxOffset;
		
		sEdgePos = gGridInfos[sCascadeId].position;
		sSpan = gGridInfos[sCascadeId].span;
	}
	__syncthreads();

	// Full Block Cull
	if(sOccupation == CSegmentOccupation::EMPTY) return;
	assert(sOccupation != CSegmentOccupation::MARKED_FOR_CLEAR);

	// Now we can cull invalid nodes
	const CVoxelNorm voxelNormPacked = gVoxelPages[pageId].dGridVoxNorm[pageLocalId];
	if(voxelNormPacked == 0xFFFFFFFF) return;

	// Find your opengl data and voxel cache
	const uint16_t& batchId = sSegInfo.batchId;
	const BatchVoxelCache& batchCache = gBatchVoxelCache[sCascadeId * batchCount + batchId];
	const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
	const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelOffset;

	// Cached Data
	const VoxelAlbedo albedoPacked = batchCache.dVoxelAlbedo[batchLocalVoxelId];

	// Required Data
	const CVoxelPos voxelPosPacked = gVoxelPages[pageId].dGridVoxPos[pageLocalId];
	const CVoxelOccupancy voxOccupPacked = gVoxelPages[pageId].dGridVoxOccupancy[pageLocalId];

	// World position reconstruction
	float3 occup = ExpandOccupancy(voxOccupPacked);
	const int3 voxPos = ExpandVoxPos(voxelPosPacked);
	float3 worldPos;
	worldPos.x = sEdgePos.x + (static_cast<float>(voxPos.x) + occup.x) * sSpan;
	worldPos.y = sEdgePos.y + (static_cast<float>(voxPos.y) + occup.y) * sSpan;
	worldPos.z = sEdgePos.z + (static_cast<float>(voxPos.z) + occup.z) * sSpan;

	// Unpack albedo and normal do the lighting calculation
	float3 unpackNormal = ExpandVoxNormal(voxelNormPacked);
	float4 unpackAlbedo = UnpackSVOIrradiance(albedoPacked);

	// Partial lighting calculation
	float3 lightDir;
	float3 irradianceDiffuse = TotalIrradiance(lightDir,
											   // Node Params
											   worldPos,
											   Normalize(unpackNormal),
											   unpackAlbedo,
											   // Light Parameters
											   liParams);
	unpackAlbedo.x = irradianceDiffuse.x;
	unpackAlbedo.y = irradianceDiffuse.y;
	unpackAlbedo.z = irradianceDiffuse.z;
	
	uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - sCascadeId;

	// Now write calculated value into nodes that are allocated
	#pragma unroll
	for(int a = 0; a < 8; a++)
	{
		uint32_t nodeLocation = gLocations[a * totalCount + nodeId];
		if(nodeLocation == 0xFFFFFFFF) return;

		char3 ijk;
		ijk.x = (a >> 0) & 0x1;
		ijk.y = (a >> 1) & 0x1;
		ijk.z = (a >> 2) & 0x1;

		// Calculte this nodes occupancy
		float3 volume;
		volume.x = (ijk.x == 1) ? occup.x : (1.0f - occup.x);
		volume.y = (ijk.y == 1) ? occup.y : (1.0f - occup.y);
		volume.z = (ijk.z == 1) ? occup.z : (1.0f - occup.z);
		float occupancy = volume.x * volume.y * volume.z;

		// Atomic Average to Location		
		CSVOIllumination* illumNode = gSVOLevels[cascadeMaxLevel].gLevelIllum + nodeLocation;

		// Irradiance Average
		AtomicIllumPortionAvg(reinterpret_cast<uint64_t*>(illumNode),
		  					  unpackAlbedo, unpackNormal);
		AtomicIllumPortionAvg(reinterpret_cast<uint64_t*>(illumNode) + 1,
		  					  unpackAlbedo, unpackNormal);
	}
}
