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


__global__ void AverageLevel(CSVOLevel& gSVOLevel,
							 const uint32_t nodeCount,
							 const OctreeParameters octreeParams)
{
	//unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	//if(globalId >= nodeCount) return;


	//#pragma unroll
	//for(int i = 0; i )
}

__global__ void ResetIllumCounter(CSVOLevel& gSVOLevel,
								  const uint32_t nodeCount)
{
	// Two Threads per load
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int illumId = globalId / 2;
	unsigned int wordId = globalId % 2;
	if(illumId >= nodeCount) return;

	// Load only required portion (compiler may load entire 128-bit illum)
	uint32_t* wordPartitionIllum = reinterpret_cast<uint32_t*>(gSVOLevel.gLevelIllum);
	uint32_t* portionWithCounter = wordPartitionIllum + illumId * 4 + wordId * 2 + 1;
	uint32_t portion = *portionWithCounter;
	if(portion != 0x0)
	{
		// Make it as if it has single node
		portion &= 0x01FFFFFF;
		*portionWithCounter = portion;


		//if(liParams.injectOn)
		//{
		//	// Gen Illumination
		//	float4 irradiance;
		//	float3 lightDir = {0.0f, 0.0f, 0.0f};

		//	// World Space Position Reconstruction
		//	const float3 edgePos = gGridInfos[cascadeId].position;
		//	const float span = gGridInfos[cascadeId].span;
		//	const uint3	voxPos = ExpandVoxPos(voxelPosPacked);
		//	float3 weights = ExpandOccupancy(voxOccupPacked);

		//	float3 worldPos;
		//	worldPos.x = edgePos.x + (static_cast<float>(voxPos.x) + weights.x) * span;
		//	worldPos.y = edgePos.y + (static_cast<float>(voxPos.y) + weights.y) * span;
		//	worldPos.z = edgePos.z + (static_cast<float>(voxPos.z) + weights.z) * span;

		//	// Normal
		//	float3 normal = ExpandVoxNormal(voxelNormPacked);

		//	// Generated Irradiance
		//	float3 irradianceDiffuse = LightInject(lightDir,
		//										   // Node Params
		//										   worldPos,
		//										   voxAlbedo,
		//										   normal,
		//										   // Light Parameters
		//										   liParams);

		//	irradiance.x = irradianceDiffuse.x;
		//	irradiance.y = irradianceDiffuse.y;
		//	irradiance.z = irradianceDiffuse.z;
		//	irradiance.w = voxAlbedo.w;

		//	irradPacked = PackSVOIrradiance(irradiance);
		//	lightDirPacked = PackVoxNormal(lightDir);
		//}
		//else
		//{
		//	irradPacked = PackSVOIrradiance(voxAlbedo);
		//}
	}	
}

__global__ void SVOReconstruct(// SVO
							   CSVOLevel* gSVOLevels,
							   uint32_t* gLevelAllocators,
							   const uint32_t* gLevelCapacities,
							   // Voxel Pages
							   const CVoxelPageConst* gVoxelPages,
							   const CVoxelGrid* gGridInfos,
							   // Cache Data (for Voxel Albedo)
							   const BatchVoxelCache* gBatchVoxelCache,
							   // Light Injection Related
							   const CLightInjectParameters liParams,
							   // Limits
							   const OctreeParameters octreeParams,
							   const uint32_t batchCount)
{
	// Shared Memory for generic data
	__shared__ CSegmentInfo sSegInfo;
	__shared__ CMeshVoxelInfo sMeshVoxelInfo;

	// Local Ids
	unsigned int blockLocalId = threadIdx.x;
	//unsigned int nodeLocalId = blockLocalId % 2;

	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GIVoxelPages::PageSize;
	unsigned int pageLocalId = globalId % GIVoxelPages::PageSize;
	unsigned int pageLocalSegmentId = pageLocalId / GIVoxelPages::SegmentSize;
	unsigned int segmentLocalVoxId = pageLocalId % GIVoxelPages::SegmentSize;

	// Get Segments Obj Information Struct
	CObjectType objType;
	CSegmentOccupation occupation;
	uint8_t cascadeId;
	bool firstOccurance;
	if(blockLocalId == 0)
	{
		// Load to smem
		// Todo split this into the threadss
		sSegInfo = gVoxelPages[pageId].dSegmentInfo[pageLocalSegmentId];
		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	}
	__syncthreads();
	if(blockLocalId != 0)
	{
		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	}
	// Full Block Cull
	if(occupation == CSegmentOccupation::EMPTY) return;
	assert(occupation != CSegmentOccupation::MARKED_FOR_CLEAR);
	if(blockLocalId == 0)
	{
		sMeshVoxelInfo = gBatchVoxelCache[cascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId];
	}
	__syncthreads();

	// Now we can cull invalid nodes
	const CVoxelNorm voxelNormPacked = gVoxelPages[pageId].dGridVoxNorm[pageLocalId];
	if(voxelNormPacked == 0xFFFFFFFF) return;

	// Unpack Vox Pos
	const CVoxelPos voxelPosPacked = gVoxelPages[pageId].dGridVoxPos[pageLocalId];

	// Unpack Occupancy
	const CVoxelOccupancy voxOccupPacked = gVoxelPages[pageId].dGridVoxOccupancy[pageLocalId];
	
	// Get Albedo (it is dependant)
	// Find your opengl data and voxel cache
	const uint16_t& batchId = sSegInfo.batchId;
	const BatchVoxelCache& batchCache = gBatchVoxelCache[cascadeId * batchCount + batchId];
	// Voxel Ids
	const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
	const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelInfo.voxOffset;
	// Voxel Albedo
	const VoxelAlbedo albedoPacked = batchCache.dVoxelAlbedo[batchLocalVoxelId];

	// Now we will start allocating all nodes
	// Each node will generate multiple neigbouring nodes (8-neigbour filtering)
	// Two threads are responsible for single node, they store the alocated positions
	// Then each thread will average upper or lower half of the node
	#pragma unroll
	for(int a = 0; a < 8; a++)
	{
		// Convert Linear Loop to 3D		
		int i = (a >> 0) & 0x1;
		int j = (a >> 1) & 0x1;
		int k = (a >> 2) & 0x1;

		const uint3	voxPos = ExpandVoxPos(voxelPosPacked);
		
		uint3 myMetaNeigbour;
		myMetaNeigbour.x = static_cast<int>(voxPos.x) + i;
		myMetaNeigbour.y = static_cast<int>(voxPos.y) + j;
		myMetaNeigbour.z = static_cast<int>(voxPos.z) + k;

		// Boundary Check
		bool validNode = (myMetaNeigbour.x < octreeParams.CascadeBaseLevelSize ||
						  myMetaNeigbour.y < octreeParams.CascadeBaseLevelSize ||
						  myMetaNeigbour.z < octreeParams.CascadeBaseLevelSize);

		const uint3 nodePos = ExpandToSVODepth(myMetaNeigbour,
											   cascadeId,
											   octreeParams.CascadeCount,
											   octreeParams.CascadeBaseLevel);

		// Allocate and Average Illumination Values
		if(validNode)
		{
			uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - cascadeId;
			CSVOIllumination* illumNode = TraverseAndAllocate(// SVO
															  gLevelAllocators,
															  gLevelCapacities,
															  gSVOLevels,
															  // Node Related
															  nodePos,
															  // Constants
															  octreeParams,
															  cascadeMaxLevel);
			
			// Calculte this nodes occupancy
			float3 weights = ExpandOccupancy(voxOccupPacked);
			float3 volume;		
			volume.x = (i == 1) ? weights.x : (1.0f - weights.x);
			volume.y = (j == 1) ? weights.y : (1.0f - weights.y);
			volume.z = (k == 1) ? weights.z : (1.0f - weights.z);
			float occupancy = volume.x * volume.y * volume.z;

			float4 unpackAlbedo = UnpackSVOIrradiance(albedoPacked);
			AtomicIllumLeafAvg(reinterpret_cast<uint64_t*>(illumNode), unpackAlbedo, occupancy);

			/*unpackAlbedo = UnpackSVONormal(voxelNormPacked);
			AtomicIllumLeafAvg(reinterpret_cast<uint64_t*>(illumNode + 1), unpackAlbedo, occupancy);*/
		}
	}

	//// Now Each Thread will load single node
	//#pragma unroll
	//for(int i = 0; i < 8; i++)
	//{
	//	uint32_t shuffleValidBits = __shfl(validBits, i / 4, 2);
	//	if((shuffleValidBits >> (i % 4)) == 0) continue;

	//	uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - cascadeId;
	//	uint32_t nodeShare = nodeLocations[i % 4];
	//	uint32_t nodeOffset = __shfl(nodeShare, i / 4, 2); 
	//	uint64_t* illumNodePartial = reinterpret_cast<uint64_t*>(gSVOLevels[cascadeMaxLevel].gLevelIllum + nodeOffset);

	//	// Calculte this nodes occupancy
	//	float3 weights = ExpandOccupancy(voxOccupPacked);
	//	float3 volume;		
	//	volume.x = (voxLookup8[i].x == 1) ? weights.x : (1.0f - weights.x);
	//	volume.y = (voxLookup8[i].y == 1) ? weights.y : (1.0f - weights.y);
	//	volume.z = (voxLookup8[i].z == 1) ? weights.z : (1.0f - weights.z);
	//	float occupancy = volume.x * volume.y * volume.z;

	//	// Portions
	//	float4 upperPortion = {0.0f};
	//	float3 lowerPortion = {0.0f};

	//	// Determine your data
	//	if(nodeLocalId == 0)
	//	{
	//		upperPortion = UnpackSVOIrradiance(irradPacked);
	//		lowerPortion = ExpandVoxNormal(voxelNormPacked);

	//		upperPortion.x *= occupancy;
	//		upperPortion.y *= occupancy;
	//		upperPortion.z *= occupancy;
	//		upperPortion.w *= occupancy;
	//	}
	//	else
	//	{
	//		// TODO: Anisotropic Occupancy
	//		upperPortion = float4{occupancy, occupancy, occupancy, occupancy};
	//		lowerPortion = ExpandVoxNormal(lightDirPacked);
	//	}
	//	lowerPortion.x *= occupancy;
	//	lowerPortion.y *= occupancy;
	//	lowerPortion.z *= occupancy;

	//	// Portion Average (Code Invariant Average)
	//	AtomicIllumPortionAvg(illumNodePartial + nodeLocalId, upperPortion, lowerPortion);
	//}
}

//
//__global__ void SVOIllumInject(// SVO
//							   CSVOLevel* gSVOLevels,
//							   uint32_t* gLevelAllocators,
//							   const uint32_t* gLevelCapacities,
//							   // Voxel Pages
//							   const CVoxelPageConst* gVoxelPages,
//							   const CVoxelGrid* gGridInfos,
//							   // Cache Data (for Voxel Albedo)
//							   const BatchVoxelCache* gBatchVoxelCache,
//							   // Light Injection Related
//							   const CLightInjectParameters liParams,
//							   // Limits
//							   const OctreeParameters octreeParams,
//							   const uint32_t batchCount)
//{
//	// Shared Memory for generic data
//	__shared__ CSegmentInfo sSegInfo;
//	__shared__ CMeshVoxelInfo sMeshVoxelInfo;
//
//	// Local Ids
//	unsigned int blockLocalId = threadIdx.x;
//	unsigned int nodeLocalId = blockLocalId % 2;
//
//	unsigned int globalId = (threadIdx.x + blockIdx.x * blockDim.x) / 2;
//	unsigned int pageId = globalId / GIVoxelPages::PageSize;
//	unsigned int pageLocalId = globalId % GIVoxelPages::PageSize;
//	unsigned int pageLocalSegmentId = pageLocalId / GIVoxelPages::SegmentSize;
//	unsigned int segmentLocalVoxId = pageLocalId % GIVoxelPages::SegmentSize;
//
//	// Get Segments Obj Information Struct
//	CObjectType objType;
//	CSegmentOccupation occupation;
//	uint8_t cascadeId;
//	bool firstOccurance;
//	if(blockLocalId == 0)
//	{
//		// Load to smem
//		// Todo split this into the threadss
//		sSegInfo = gVoxelPages[pageId].dSegmentInfo[pageLocalSegmentId];
//		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
//	}
//	__syncthreads();
//	if(blockLocalId != 0)
//	{
//		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
//	}
//	// Full Block Cull
//	if(occupation == CSegmentOccupation::EMPTY) return;
//	assert(occupation != CSegmentOccupation::MARKED_FOR_CLEAR);
//	if(blockLocalId == 0)
//	{
//		sMeshVoxelInfo = gBatchVoxelCache[cascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId];
//	}
//	__syncthreads();
//
//	// Fetch Position and Normal
//	// Generate Light Direction and Irradiance
//	const CVoxelPos voxelPosPacked = gVoxelPages[pageId].dGridVoxPos[pageLocalId];
//	const CVoxelNorm voxelNormPacked = gVoxelPages[pageId].dGridVoxNorm[pageLocalId];
//
//	// Unpack Occupancy
//	const CVoxelOccupancy voxOccupPacked = gVoxelPages[pageId].dGridVoxOccupancy[pageLocalId];
//	float3 weights = ExpandOccupancy(voxOccupPacked);
//
//	// Now we can cull invalid nodes
//	if(voxelNormPacked == 0xFFFFFFFF) return;
//
//	// Illum Data Packed
//	VoxelAlbedo irradPacked = 0;
//	VoxelNormal lightDirPacked = 0;
//
//	//// Light Injection
//	//if(nodeLocalId == 0)
//	//{
//	//	// Now load albeo etc and average those on leaf levels
//	//	// Find your opengl data and voxel cache
//	//	const uint16_t& batchId = sSegInfo.batchId;
//	//	const BatchVoxelCache& batchCache = gBatchVoxelCache[cascadeId * batchCount + batchId];
//
//	//	// Voxel Ids
//	//	const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
//	//	const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelInfo.voxOffset;
//
//	//	// Voxel Albedo
//	//	const CVoxelAlbedo voxAlbedoPacked = batchCache.dVoxelAlbedo[batchLocalVoxelId];
//	//	float4 voxAlbedo = UnpackSVOIrradiance(*reinterpret_cast<const CSVOIrradiance*>(&voxAlbedoPacked));
//
//	//	if(liParams.injectOn)
//	//	{
//	//		// Gen Illumination
//	//		float4 irradiance;
//	//		float3 lightDir = {0.0f, 0.0f, 0.0f};
//
//	//		// World Space Position Reconstruction
//	//		const float3 edgePos = gGridInfos[cascadeId].position;
//	//		const float span = gGridInfos[cascadeId].span;
//	//		const uint3	voxPos = ExpandVoxPos(voxelPosPacked);
//
//	//		float3 worldPos;
//	//		worldPos.x = edgePos.x + (static_cast<float>(voxPos.x) + weights.x) * span;
//	//		worldPos.y = edgePos.y + (static_cast<float>(voxPos.y) + weights.y) * span;
//	//		worldPos.z = edgePos.z + (static_cast<float>(voxPos.z) + weights.z) * span;
//
//	//		// Normal
//	//		float3 normal = ExpandVoxNormal(voxelNormPacked);
//
//	//		// Generated Irradiance
//	//		float3 irradianceDiffuse = LightInject(lightDir,
//	//											   // Node Params
//	//											   worldPos,
//	//											   voxAlbedo,
//	//											   normal,
//	//											   // Light Parameters
//	//											   liParams);
//
//	//		irradiance.x = irradianceDiffuse.x;
//	//		irradiance.y = irradianceDiffuse.y;
//	//		irradiance.z = irradianceDiffuse.z;
//	//		irradiance.w = voxAlbedo.w;
//
//	//		irradPacked = PackSVOIrradiance(irradiance);
//	//		lightDirPacked = PackVoxNormal(lightDir);
//	//	}
//	//	else
//	//	{
//	//		irradPacked = PackSVOIrradiance(voxAlbedo);
//	//	}
//	//}
//	// Transfer illum data to neigbour
//	//unsigned int warpLocalNodeId = globalId % (warpSize >> 1);
//	irradPacked = __shfl(irradPacked, 0, 2);
//	lightDirPacked = __shfl(lightDirPacked, 0, 2);
//
//	// Now we will start allocating all nodes
//	// Each node will generate multiple neigbouring nodes (8-neigbour filtering)
//	// Two threads are responsible for single node, they store the alocated positions
//	// Then each thread will average upper or lower half of the node
//	uint4 nodeLocations; uint8_t validBits = 0x00;
//	#pragma unroll
//	for(int j = 0; j < 4; j++)
//	{
//		const uint3	voxPos = ExpandVoxPos(voxelPosPacked);
//
//		int3 myMetaNeigbour;
//		myMetaNeigbour.x = static_cast<int>(voxPos.x) + voxLookup8[nodeLocalId * 4 + j].x;
//		myMetaNeigbour.y = static_cast<int>(voxPos.y) + voxLookup8[nodeLocalId * 4 + j].y;
//		myMetaNeigbour.z = static_cast<int>(voxPos.z) + voxLookup8[nodeLocalId * 4 + j].z;
//
//		// Boundary Check
//		bool validNode = (myMetaNeigbour.x >= 0 || myMetaNeigbour.x < octreeParams.CascadeBaseLevelSize ||
//						  myMetaNeigbour.y >= 0 || myMetaNeigbour.y < octreeParams.CascadeBaseLevelSize ||
//						  myMetaNeigbour.z >= 0 || myMetaNeigbour.z < octreeParams.CascadeBaseLevelSize);
//		validBits |= ((validNode) ? 1 : 0) << j;
//
//		uint3 uMyMetaNeigbour;
//		uMyMetaNeigbour.x = static_cast<unsigned int>(myMetaNeigbour.x);
//		uMyMetaNeigbour.y = static_cast<unsigned int>(myMetaNeigbour.y);
//		uMyMetaNeigbour.z = static_cast<unsigned int>(myMetaNeigbour.z);
//		const uint3 nodePos = ExpandToSVODepth(uMyMetaNeigbour,
//											   cascadeId,
//											   octreeParams.CascadeCount,
//											   octreeParams.CascadeBaseLevel);
//
//		// Allocate and Average Illumination Values
//		if(validNode)
//		{
//			uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - cascadeId;
//			uint32_t test;
//			const CSVONode* illumNode = TraverseNode(test,
//													  // SVO
//													  reinterpret_cast<const CSVOLevelConst*>(gSVOLevels),
//													  // Node Related
//													  nodePos,
//													  // Constants
//													  octreeParams,
//													  cascadeMaxLevel);
//
//			uint32_t nodeOffset = illumNode - gSVOLevels[cascadeMaxLevel].gLevelNodes;
//			reinterpret_cast<unsigned int*>(&nodeLocations)[j] = nodeOffset;
//		}
//	}
//
//	// Now Each Thread will load single node
//	#pragma unroll
//	for(int i = 0; i < 8; i++)
//	{
//		uint32_t shuffleValidBits = __shfl(validBits, (i < 4) ? 0 : 1, 2);
//		if((shuffleValidBits >> (i % 4)) == 0) continue;
//
//		uint32_t cascadeMaxLevel = octreeParams.MaxSVOLevel - cascadeId;
//		uint32_t nodeShare = reinterpret_cast<unsigned int*>(&nodeLocations)[i % 4];
//		uint32_t nodeOffset = __shfl(nodeShare, (i < 4) ? 0 : 1, 2); 
//		uint64_t* illumNodePartial = reinterpret_cast<uint64_t*>(gSVOLevels[cascadeMaxLevel].gLevelIllum + nodeOffset);
//
//		float4 upperPortion = {0.0f};
//		float3 lowerPortion = {0.0f};
//
//		// Calculte this nodes occupancy
//		float3 volume;
//		volume.x = (voxLookup8[i].x == 1) ? weights.x : (1.0f - weights.x);
//		volume.y = (voxLookup8[i].y == 1) ? weights.y : (1.0f - weights.y);
//		volume.z = (voxLookup8[i].z == 1) ? weights.z : (1.0f - weights.z);
//		float occupancy = 0; volume.x * volume.y * volume.z;
//
//		// Determine your data
//		if(nodeLocalId == 0)
//		{
//			upperPortion = UnpackSVOIrradiance(irradPacked);
//			lowerPortion = ExpandVoxNormal(voxelNormPacked);
//
//			upperPortion.x *= occupancy;
//			upperPortion.y *= occupancy;
//			upperPortion.z *= occupancy;
//			upperPortion.w *= occupancy;
//		}
//		else
//		{
//			// TODO: Anisotropic Occupancy
//			upperPortion = float4{occupancy, occupancy, occupancy, occupancy};
//			lowerPortion = ExpandVoxNormal(lightDirPacked);
//		}
//		lowerPortion.x *= occupancy;
//		lowerPortion.y *= occupancy;
//		lowerPortion.z *= occupancy;
//
//		// Portion Average (Code Invariant Average)
//		AtomicIllumPortionAvg(illumNodePartial + nodeLocalId, upperPortion, lowerPortion);
//	}
//}