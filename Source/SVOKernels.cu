#include "SVOKernels.cuh"
#include "GISparseVoxelOctree.h"
#include "GIVoxelPages.h"
#include "CSVOHash.cuh"
#include "CVoxelFunctions.cuh"
#include "CSVOLightInject.cuh"
#include "CSVOIllumAverage.cuh"
#include "CSVONodeAlloc.cuh"
#include <cuda.h>

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

	const uint32_t nodeChildrenBase = gCurrentLevel.gLevelNodes[nodeId].next;

	// Each Thread will average (and read/write)
	// single double word part of the illumination
	// Average Children
	if(nodeChildrenBase != 0xFFFFFFFF)
	{
		uint64_t illumPart = 0x0;

		// Read potential parent value		
		if(isCascadeLevel) illumPart = reinterpret_cast<uint64_t*>(gCurrentLevel.gLevelIllum + nodeId)[nodeLocalId];

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

//__global__ void GenFrontNeighborPtrs(// SVO
//									 const CSVOLevel* gSVOLevels,
//									 uint32_t* gLevelAllocators,
//									 const uint32_t* gLevelCapacities,
//									 // Limits
//									 const OctreeParameters octreeParams,
//									 const uint32_t nodeCount,
//									 const uint32_t level)
//{
//	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	if(globalId >= nodeCount) return;
//
//	const CSVOLevel& currentLevel = gSVOLevels[level];
//	uint32_t voxPosPacked = currentLevel.gVoxId[globalId];
//
//	if(voxPosPacked != 0xFFFFFFFF)
//	{
//		uint32_t cascadeId = octreeParams.MaxSVOLevel - level;
//		int3 nodePos = ExpandToSVODepth(ExpandVoxPos(voxPosPacked),
//										cascadeId,
//										octreeParams.CascadeCount,
//										octreeParams.CascadeBaseLevel);
//		int levelId = (0x1 << level);
//		
//		// Force gen back neigbours		
//		nodePos.x += 1;
//		if(nodePos.x < levelId)
//		{
//			uint32_t traversedLevel;
//			uint32_t nodeLocation = TraverseNode(traversedLevel,
//												 reinterpret_cast<const CSVOLevelConst*>(gSVOLevels),
//												 nodePos, octreeParams, level);
//			if(traversedLevel == level) 
//				gSVOLevels[level].gLevelNodes[globalId].neigbours[0] = nodeLocation;
//			else
//			{
//				// Here is corner special case
//				// if x->y->z top right corner neighbor has illum value (and none other has illum)
//				// we cant traverse from x->y->z (or any other combination)
//				// we need to solve this case 
//				int3 cornerPos = nodePos;
//				cornerPos.y += 1;
//				cornerPos.z += 1;				
//				uint32_t cornerLoc = TraverseNode(traversedLevel,
//												  reinterpret_cast<const CSVOLevelConst*>(gSVOLevels),
//												  nodePos, octreeParams, level);
//				
//				const uint32_t* cornerIllumLoc = reinterpret_cast<const uint32_t*>(gSVOLevels[traversedLevel].gLevelIllum + cornerLoc);
//				if(traversedLevel == level && cornerIllumLoc[0] == 0x0)
//				{
//					// Corner has value and we couldnt generate a x neighbor node
//					// Force forward gen 
//					uint32_t xNLoc = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
//													  nodePos, octreeParams, level, false);
//					gSVOLevels[level].gLevelNodes[globalId].neigbours[0] = xNLoc;
//
//					// And Link X neigbours y neighbor (that y neighbor is available)
//					nodePos.y += 1;
//					uint32_t xyNLoc = TraverseNode(traversedLevel,
//												   reinterpret_cast<const CSVOLevelConst*>(gSVOLevels),
//												   nodePos, octreeParams, level);
//					assert(traversedLevel == level);
//					gSVOLevels[level].gLevelNodes[xyNLoc].neigbours[1] = xNLoc;
//					nodePos.y -= 1;
//				}
//			}
//		}
//		nodePos.x -= 1;
//		nodePos.y += 1;
//		if(nodePos.y < levelId)
//		{
//			uint32_t traversedLevel;
//			uint32_t nodeLocation = TraverseNode(traversedLevel,
//												 reinterpret_cast<const CSVOLevelConst*>(gSVOLevels),
//												 nodePos, octreeParams, level);
//			if(traversedLevel == level)
//				gSVOLevels[level].gLevelNodes[globalId].neigbours[1] = nodeLocation;
//		}
//		nodePos.y -= 1;
//		nodePos.z += 1;
//		if(nodePos.z < levelId)
//		{
//			uint32_t traversedLevel;
//			uint32_t nodeLocation = TraverseNode(traversedLevel,
//												 reinterpret_cast<const CSVOLevelConst*>(gSVOLevels),
//												 nodePos, octreeParams, level);
//			if(traversedLevel == level) 
//				gSVOLevels[level].gLevelNodes[globalId].neigbours[2] = nodeLocation;
//		}
//		currentLevel.gVoxId[globalId] = 0xFFFFFFFF;
//	}
//}

__global__ void GenBackNeighborPtrs(// SVO
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
	uint32_t voxPosPacked = currentLevel.gVoxId[globalId];
	if(voxPosPacked != 0xFFFFFFFF)
	{
		uint32_t cascadeId = octreeParams.MaxSVOLevel - level;
		int3 nodePos = ExpandToSVODepth(ExpandVoxPos(voxPosPacked),
										cascadeId,
										octreeParams.CascadeCount,
										octreeParams.CascadeBaseLevel);

		uint32_t nodeLocations[7];
		nodeLocations[0] = 0xFFFFFFFF;
		nodeLocations[1] = 0xFFFFFFFF;
		nodeLocations[2] = 0xFFFFFFFF;
		nodeLocations[3] = 0xFFFFFFFF;
		nodeLocations[4] = 0xFFFFFFFF;
		nodeLocations[5] = 0xFFFFFFFF;
		nodeLocations[6] = 0xFFFFFFFF;

		// Force gen back neigbours (1st order)
		nodePos.x -= 1;
		if(nodePos.x >= 0)
		{
			nodeLocations[0] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
												nodePos, octreeParams, level, false);
			gSVOLevels[level].gLevelNodes[nodeLocations[0]].neigbours[0] = globalId;
		}
		nodePos.x += 1;
		nodePos.y -= 1;
		if(nodePos.y >= 0)
		{
			nodeLocations[1] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
													 nodePos, octreeParams, level, false);
			gSVOLevels[level].gLevelNodes[nodeLocations[1]].neigbours[1] = globalId;
		}
		nodePos.y += 1;
		nodePos.z -= 1;
		if(nodePos.z >= 0)
		{
			nodeLocations[2] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
													 nodePos, octreeParams, level, false);
			gSVOLevels[level].gLevelNodes[nodeLocations[2]].neigbours[2] = globalId;
		}		

		// 2nd order
		nodePos.z += 1;
		nodePos.x -= 1;
		nodePos.y -= 1;
		if(nodePos.x >= 0 && nodePos.y >= 0)
		{
			nodeLocations[3] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
													 nodePos, octreeParams, level, false);
			gSVOLevels[level].gLevelNodes[nodeLocations[3]].neigbours[0] = nodeLocations[1];
			gSVOLevels[level].gLevelNodes[nodeLocations[3]].neigbours[1] = nodeLocations[0];
		}
		nodePos.x += 1;
		nodePos.z -= 1;
		if(nodePos.y >= 0 && nodePos.z >= 0)
		{
			nodeLocations[4] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
													 nodePos, octreeParams, level, false);

			gSVOLevels[level].gLevelNodes[nodeLocations[4]].neigbours[1] = nodeLocations[2];
			gSVOLevels[level].gLevelNodes[nodeLocations[4]].neigbours[2] = nodeLocations[1];
		}
		nodePos.y += 1;
		nodePos.x -= 1;
		if(nodePos.x >= 0 && nodePos.z >= 0)
		{
			nodeLocations[5] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
													 nodePos, octreeParams, level, false);

			gSVOLevels[level].gLevelNodes[nodeLocations[5]].neigbours[0] = nodeLocations[2];
			gSVOLevels[level].gLevelNodes[nodeLocations[5]].neigbours[2] = nodeLocations[0];
		}

		// Third order
		nodePos.y -= 1;
		if(nodePos.x >= 0 && nodePos.y >= 0 && nodePos.z >= 0)
		{
			nodeLocations[6] = PunchThroughNode(gLevelAllocators, gLevelCapacities, gSVOLevels,
													 nodePos, octreeParams, level, false);

			gSVOLevels[level].gLevelNodes[nodeLocations[6]].neigbours[0] = nodeLocations[4];
			gSVOLevels[level].gLevelNodes[nodeLocations[6]].neigbours[1] = nodeLocations[5];
			gSVOLevels[level].gLevelNodes[nodeLocations[6]].neigbours[2] = nodeLocations[3];
		}

		currentLevel.gVoxId[globalId] = 0xFFFFFFFF;
	}
}

__global__ void AdjustIllumParameters(const CSVOLevel& gSVOLevel, uint32_t nodeCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= nodeCount) return;

	uint32_t* myIllumPtr = reinterpret_cast<uint32_t*>(gSVOLevel.gLevelIllum + globalId);

	// Load illum data and world pos
	uint32_t illum1 = myIllumPtr[1];
	uint32_t illum2 = myIllumPtr[2];
	
	if(illum1 != 0x0)
	{
		// Illum0 already laid out well
		float3 unpack1 = UnpackSVOUpperLeaf(illum1);
		float4 unpack2 = UnpackLightDirLeaf(illum2);
		float occlusion = fminf(unpack1.z, 1.0f);

		myIllumPtr[1] = PackSVONormal(float4{unpack1.x, unpack1.y, unpack2.w, 1.0f});
		myIllumPtr[2] = PackSVOIrradiance(float4{occlusion, occlusion, occlusion, occlusion});
		myIllumPtr[3] = PackSVOLightDir(float4{unpack2.x, unpack2.y, unpack2.z, 1.0f});
	}
}

__global__ void SVOReconstruct(// SVO
							   const CSVOLevel* gSVOLevels,
							   uint32_t* gLevelAllocators,
							   const uint32_t* gLevelCapacities,
							   // Voxel Pages
							   const CVoxelPageConst* gVoxelPages,
							   const CVoxelGrid* gGridInfos,
							   // Cache Data (for Voxel Albedo)
							   const BatchVoxelCache* gBatchVoxelCache,
							   // Inject Related
							   const CLightInjectParameters liParams,
							   // Limits			
							   const OctreeParameters octreeParams,
							   const uint32_t batchCount)
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
	float3 lightDir = {0.0f, 0.0f, 0.0f};
	if(liParams.injectOn)
	{
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
	}

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

		float2 normalXY;
		float4 lightDirXYZNormalW;
		normalXY.x = unpackNormal.x;
		normalXY.y = unpackNormal.y;
		lightDirXYZNormalW.x = lightDir.x;
		lightDirXYZNormalW.y = lightDir.y;
		lightDirXYZNormalW.z = lightDir.z;
		lightDirXYZNormalW.w = unpackNormal.z;

		// Allocate and Average Illumination Values
		if(validNode)
		{
			uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - sCascadeId;

			uint32_t nodeLocation = PunchThroughNode(// SVO
													 gLevelAllocators,
													 gLevelCapacities,
													 gSVOLevels,
													 // Node Related
													 nodePos,
													 // Constants
													 octreeParams,
													 cascadeMaxLevel,
													 true);

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
								  unpackAlbedo, normalXY, occupancy);
			AtomicIllumPortionAvg(reinterpret_cast<uint64_t*>(illumNode) + 1,
								  lightDirXYZNormalW, occupancy);
		}
	}
}