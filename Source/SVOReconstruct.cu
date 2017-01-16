#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CVoxel.cuh"
#include "SVOLightInject.cuh"
#include <cuda.h>

// Lookup table for determining neigbour nodes
// just splitted first 8 values
__device__ static const char3 voxLookup[] =
{
	{0, 0, 0},
	{1, 0, 0},
	{0, 1, 0},
	{1, 1, 0},

	{0, 0, 1},
	{1, 0, 1},
	{0, 1, 1},
	{1, 1, 1}
};

inline __device__ uint64_t AverageColorNormal(const uint64_t& material,
											  const float4& colorUnpack,
											  const float3& normalUnpack)
{
    // Unpack Material
    CSVOColor avgColorPacked;
    CVoxelNorm avgNormalPacked;
    UnpackSVOMaterialColorNormal(avgColorPacked, avgNormalPacked, material);
    float4 avgColor = UnpackSVOColor(avgColorPacked);
    float4 avgNormal = UnpackSVONormal(avgNormalPacked);

    // Averaging (color.w is number of nodes)
    //assert(avgNormal.w <= 255.0f);
    if(avgNormal.w <= 255.0f);
    {
        float ratio = avgNormal.w / (avgNormal.w + 1.0f);

        // New Color Average
        avgColor.x = (ratio * avgColor.x) + (colorUnpack.x / (avgNormal.w + 1.0f));
        avgColor.y = (ratio * avgColor.y) + (colorUnpack.y / (avgNormal.w + 1.0f));
        avgColor.z = (ratio * avgColor.z) + (colorUnpack.z / (avgNormal.w + 1.0f));
        avgColor.w = (ratio * avgColor.w) + (colorUnpack.w / (avgNormal.w + 1.0f));

        // New Normal Average
        avgNormal.x = (ratio * avgNormal.x) + (normalUnpack.x / (avgNormal.w + 1.0f));
        avgNormal.y = (ratio * avgNormal.y) + (normalUnpack.y / (avgNormal.w + 1.0f));
        avgNormal.z = (ratio * avgNormal.z) + (normalUnpack.z / (avgNormal.w + 1.0f));
        avgNormal.w += 1.0f;
    }
    avgColorPacked = PackSVOColor(avgColor);
    avgNormalPacked = PackSVONormal(avgNormal);
    return PackSVOMaterialColorNormal(avgColorPacked, avgNormalPacked);
}

inline __device__ uint64_t AverageAniso(const uint64_t& aniso,
										const float4& anisoXY,
										const float2& anisoZ)
{
	// Unpack Material
	CSVOAnisoXY avgAnisoXYPacked;
	CSVOAnisoZ avgAnisoZPacked;
	UnpackSVOAnisoOccupancy(avgAnisoXYPacked, avgAnisoZPacked, aniso);
	float4 avgAnisoXY = UnpackSVOAnisoXY(avgAnisoXYPacked);
	float4 avgAnisoZ = UnpackSVOAnisoZ(avgAnisoZPacked);

	// Averaging (color.w is number of nodes)
	//assert(avgNormal.w <= 255.0f);
	avgAnisoXY.x += anisoXY.x;
	avgAnisoXY.y += anisoXY.y;
	avgAnisoXY.z += anisoXY.z;
	avgAnisoXY.w += anisoXY.w;
	avgAnisoZ.x  += anisoZ.x;
	avgAnisoZ.y  += anisoZ.y;

	avgAnisoXY.x = fminf(avgAnisoXY.x, 1.0f);
	avgAnisoXY.y = fminf(avgAnisoXY.y, 1.0f);
	avgAnisoXY.z = fminf(avgAnisoXY.z, 1.0f);
	avgAnisoXY.w = fminf(avgAnisoXY.w, 1.0f);
	avgAnisoZ.x  = fminf(avgAnisoZ.x, 1.0f);
	avgAnisoZ.y  = fminf(avgAnisoZ.y, 1.0f);
	
	//if(avgAnisoZ.w <= 255.0f);
	//{
	//	float ratio = avgAnisoZ.w / (avgAnisoZ.w + 1.0f);
	//	// XY
	//	avgAnisoXY.x = (ratio * avgAnisoXY.x) + (anisoXY.x / (avgAnisoZ.w + 1.0f));
	//	avgAnisoXY.y = (ratio * avgAnisoXY.y) + (anisoXY.y / (avgAnisoZ.w + 1.0f));
	//	avgAnisoXY.z = (ratio * avgAnisoXY.z) + (anisoXY.z / (avgAnisoZ.w + 1.0f));
	//	avgAnisoXY.w = (ratio * avgAnisoXY.w) + (anisoXY.w / (avgAnisoZ.w + 1.0f));

	//	// Z
	//	avgAnisoZ.x = (ratio * avgAnisoZ.x) + (anisoZ.x / (avgAnisoZ.w + 1.0f));
	//	avgAnisoZ.y = (ratio * avgAnisoZ.y) + (anisoZ.y / (avgAnisoZ.w + 1.0f));
	//	//avgAnisoZ.z = (ratio * avgAnisoZ.z) + (anisoZ.z / (avgAnisoZ.w + 1.0f));
	//	avgAnisoZ.w += 1.0f;
	//}
	avgAnisoXYPacked = PackAnisoXY(avgAnisoXY);
	avgAnisoZPacked = PackAnisoZ(avgAnisoZ);
	return PackSVOAnisoOccupancy(avgAnisoXYPacked, avgAnisoZPacked);
}


//inline __device__ CSVOMaterial AddMat(const CSVOMaterial& material,
//                                      const float4& colorUnpack,
//                                      const float4& normalUnpack)
//{
//    // Unpack Material
//    CSVOColor avgColorPacked;
//    CVoxelNorm avgNormalPacked;
//    UnpackSVOMaterial(avgColorPacked, avgNormalPacked, material);
//    float4 avgColor = UnpackSVOColor(avgColorPacked);
//    float4 avgNormal = ExpandOnlyNormal(avgNormalPacked);
//
//    // Accum Color
//    avgColor.x += colorUnpack.x;
//    avgColor.y += colorUnpack.y;
//    avgColor.z += colorUnpack.z;
//    avgColor.w += colorUnpack.w;
//
//    // New Normal Average
//    avgNormal.x += normalUnpack.x;
//    avgNormal.y += normalUnpack.y;
//    avgNormal.z += normalUnpack.z;
//    avgNormal.w += normalUnpack.w;
//
//    avgColorPacked = PackSVOColor(avgColor);
//    avgNormalPacked = PackOnlyVoxNorm(avgNormal);
//    return PackSVOMaterial(avgColorPacked, avgNormalPacked);
//}

inline __device__ uint64_t AtomicAnisoAvg(uint64_t* gMaterialAniso,
										  const CSVOAnisoXY& anisoXY,
										  const CSVOAnisoZ& anisoZ)
{
	float4 anisoXYUnpack = UnpackSVOAnisoXY(anisoXY);
	float4 anisoZUnpack = UnpackSVOAnisoZ(anisoZ);
	uint64_t assumed, old = *gMaterialAniso;
	do
	{
		assumed = old;
		old = atomicCAS(gMaterialAniso, assumed,
						AverageAniso(assumed,
									 anisoXYUnpack,
									 {anisoZUnpack.x, anisoZUnpack.y}));
	} while(assumed != old);
	return old;
}

inline __device__ uint64_t AtomicColorNormalAvg(uint64_t* gMaterialColorNormal,
												const CSVOColor& color,
												const CVoxelNorm& voxelNormal,
												float occupancy)
{
	float4 colorUnpack = UnpackSVOColor(color);
	float4 normalUnpack = UnpackSVONormal(voxelNormal);
	uint64_t assumed, old = *gMaterialColorNormal;

	//colorUnpack.x *= occupancy;
	//colorUnpack.y *= occupancy;
	//colorUnpack.z *= occupancy;
	//colorUnpack.w *= occupancy;

	//normalUnpack.x *= occupancy;
	//normalUnpack.y *= occupancy;
	//normalUnpack.z *= occupancy;
	//normalUnpack.w *= occupancy;

	do
	{
		assumed = old;
		old = atomicCAS(gMaterialColorNormal, assumed,
						AverageColorNormal(assumed,
										   colorUnpack,
										   {normalUnpack.x, normalUnpack.y, normalUnpack.z}));
	} while(assumed != old);
	return old;
}

inline __device__ CSVOMaterial AtomicAvg(CSVOMaterial* gMaterial,
										 const CSVOColor& color,
										 const CVoxelNorm& voxelNormal,
										 const float& occupancy)
{
	uint64_t avgCN = AtomicColorNormalAvg(&(gMaterial->normalColor), color, voxelNormal, occupancy);

	CSVOAnisoXY anisoXY = PackAnisoXY({occupancy, occupancy, occupancy, occupancy});
	CSVOAnisoZ anisoZ = PackAnisoXY({occupancy, occupancy, 0.0f, 0.0f});
	uint64_t avgA = AtomicAnisoAvg(&(gMaterial->anisoOccupancy), anisoXY, anisoZ);
	//gMaterial->anisoOccupancy = 0x0000FFFFFFFFFFFF;
	//return CSVOMaterial{avgCN, 0x0000FFFFFFFFFFFF};
	return CSVOMaterial{avgCN, avgA};
}

//inline __device__ uint64_t AtomicColorNormalAvg(uint64_t* gMaterialColorNormal,
//												const CSVOColor& color,
//												const CVoxelNorm& voxelNormal)
//{
//	float4 colorUnpack = UnpackSVOColor(color);
//	float4 normalUnpack = ExpandOnlyNormal(voxelNormal);
//	uint64_t assumed, old = *gMaterialColorNormal;
//	do
//	{
//		assumed = old;
//		old = atomicCAS(gMaterialColorNormal, assumed,
//						AverageColorNormal(assumed,
//										   colorUnpack,
//										   {normalUnpack.x, normalUnpack.y, normalUnpack.z}));
//	} while(assumed != old);
//	return old;
//}


//inline __device__ CSVOMaterial AtomicMatAdd(CSVOMaterial* gMaterial,
//                                            const CSVOColor& color,
//                                            const CVoxelNorm& voxelNormal)
//{
//    float4 colorUnpack = UnpackSVOColor(color);
//    float4 normalUnpack = ExpandOnlyNormal(voxelNormal);
//    CSVOMaterial assumed, old = *gMaterial;
//    do
//    {
//        assumed = old;
//        old = atomicCAS(gMaterial, assumed, AddMat(assumed,
//                                                   colorUnpack,
//                                                   normalUnpack));
//    } while(assumed != old);
//    return old;
//}

inline __device__ unsigned int AtomicAllocateNode(CSVONode* gNode, unsigned int& gLevelAllocator)
{
    // Release Configuration Optimization fucks up the code
    // Prob changes some memory i-o ordering
    // Its fixed but comment is here for future
    // Problem here was cople threads on the same warp waits eachother and
    // after some memory ordering changes by compiler responsible thread waits
    // other threads execution to be done
    // Code becomes something like this after compiler changes some memory orderings
    //
    //	while(old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE) == 0xFFFFFFFE); <-- notice semicolon
    //	 if(old == 0xFFFFFF)
    //		location = allocate();
    //	location = old;
    //	return location;
    //
    // first allocating thread will never return from that loop since 
    // its warp threads are on infinite loop (so deadlock)

    // much cooler version can be warp level exchange intrinsics
    // which slightly reduces atomic pressure on the single node (on lower tree levels atleast)
    if(*gNode < 0xFFFFFFFE) return *gNode;

    CSVONode old = 0xFFFFFFFE;
    while(old == 0xFFFFFFFE)
    {
        old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE);
        if(old == 0xFFFFFFFF)
        {
            // Allocate
            unsigned int location = atomicAdd(&gLevelAllocator, 8);
            *reinterpret_cast<volatile CSVONode*>(gNode) = location;
            old = location;
        }
        __threadfence();	// This is important somehow compiler changes this and makes infinite loop on same warp threads
    }
    return old;
}

inline __device__ unsigned int FindDenseChildren(const uint3& parentIndex,
                                                 const unsigned int childId,
                                                 const unsigned int levelDim)
{
    // Go down 1 lvl
    uint3 childIndex = parentIndex;
    childIndex.x *= 2;
    childIndex.y *= 2;
    childIndex.z *= 2;

    uint3 offsetIndex =
    {
        childId % 2,
        childId / 2,
        childId / 4
    };
    childIndex.x += offsetIndex.x;
    childIndex.y += offsetIndex.y;
    childIndex.z += offsetIndex.z;

    unsigned int childLvlDim = levelDim << 1;
    unsigned int linearChildId = childIndex.z * childLvlDim * childLvlDim +
        childIndex.y * childLvlDim +
        childIndex.z;
    return linearChildId;
}

__global__ void SVOReconstructDetermineNode(CSVONode* gSVODense,
                                            const CVoxelPage* gVoxelData,

                                            const unsigned int cascadeNo,
                                            const CSVOConstants& svoConstants)
{
    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pageId = globalId / GI_PAGE_SIZE;
    unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
    unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

    // Skip Whole segment if necessary
    if(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::EMPTY) return;
    assert(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) != SegmentOccupation::MARKED_FOR_CLEAR);

    // Fetch voxel
    CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
    if(voxelPosPacked == 0xFFFFFFFF) return;

    // Local Voxel pos and expand it if its one of the inner cascades
    uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
    uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
                                      svoConstants.numCascades,
                                      svoConstants.totalDepth);
    uint3 denseIndex = CalculateLevelVoxId(voxelPos, svoConstants.denseDepth,
                                           svoConstants.totalDepth);

    assert(denseIndex.x < svoConstants.denseDim &&
           denseIndex.y < svoConstants.denseDim &&
           denseIndex.z < svoConstants.denseDim);

    // Signal alloc
    *(gSVODense +
      svoConstants.denseDim * svoConstants.denseDim * denseIndex.z +
      svoConstants.denseDim * denseIndex.y +
      denseIndex.x) = 1;
}

__global__ void SVOReconstructDetermineNode(CSVONode* gSVOSparse,
                                            cudaTextureObject_t tSVODense,
                                            const CVoxelPage* gVoxelData,
                                            const unsigned int* gLevelOffsets,

                                            // Constants
                                            const unsigned int cascadeNo,
                                            const unsigned int levelDepth,
                                            const CSVOConstants& svoConstants)
{
    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pageId = globalId / GI_PAGE_SIZE;
    unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
    unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

    // Skip Whole segment if necessary
    if(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::EMPTY) return;
    assert(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) != SegmentOccupation::MARKED_FOR_CLEAR);

    // Fetch voxel
    CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
    if(voxelPosPacked == 0xFFFFFFFF) return;

    // Local Voxel pos and expand it if its one of the inner cascades
    uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
    uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
                                      svoConstants.numCascades,
                                      svoConstants.totalDepth);

    unsigned int nodeIndex = 0;
    for(unsigned int i = svoConstants.denseDepth; i < levelDepth; i++)
    {
        CSVONode currentNode;
        if(i == svoConstants.denseDepth)
        {
            uint3 denseIndex = CalculateLevelVoxId(voxelPos, svoConstants.denseDepth,
                                                   svoConstants.totalDepth);

            assert(denseIndex.x < svoConstants.denseDim &&
                   denseIndex.y < svoConstants.denseDim &&
                   denseIndex.z < svoConstants.denseDim);

            currentNode = tex3D<unsigned int>(tSVODense,
                                              denseIndex.x,
                                              denseIndex.y,
                                              denseIndex.z);
        }
        else
        {
            currentNode = gSVOSparse[gLevelOffsets[i - svoConstants.denseDepth] + nodeIndex];
        }

        // Offset according to children
        assert(currentNode != 0xFFFFFFFF);
        unsigned int childIndex = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
        nodeIndex = currentNode + childIndex;
    }

    // Finally Write
    gSVOSparse[gLevelOffsets[levelDepth - svoConstants.denseDepth] + nodeIndex] = 1;
}

__global__ void SVOReconstructAllocateLevel(CSVONode* gSVOLevel,
                                            unsigned int& gSVONextLevelAllocator,
                                            const unsigned int& gSVONextLevelTotalSize,
                                            const unsigned int& gSVOLevelSize,
                                            const CSVOConstants& svoConstants)
{
    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    if(globalId >= gSVOLevelSize) return;

    CSVONode node = gSVOLevel[globalId]; if(node != 1) return;

    // Allocation
    unsigned int location = atomicAdd(&gSVONextLevelAllocator, 8);
    assert(location < gSVONextLevelTotalSize);

    gSVOLevel[globalId] = location;
}

__global__ void SVOReconstructMaterialLeaf(CSVOMaterial* gSVOMat,

                                           // Const SVO Data
                                           const CSVONode* gSVOSparse,
                                           const unsigned int* gLevelOffsets,
                                           cudaTextureObject_t tSVODense,

                                           // Page Data
                                           const CVoxelPage* gVoxelData,

                                           // For Color Lookup
                                           CVoxelColor** gVoxelRenderData,

                                           // Constants
                                           const unsigned int matSparseOffset,
                                           const unsigned int cascadeNo,
                                           const CSVOConstants& svoConstants,

                                           // Light Inject Related
                                           bool inject,
                                           float span,
                                           const float3 outerCascadePos,
                                           const float3 ambientColor,

                                           const float4 camPos,
                                           const float3 camDir,

                                           const CMatrix4x4* lightVP,
                                           const CLight* lightStruct,

                                           const float depthNear,
                                           const float depthFar,

                                           cudaTextureObject_t shadowMaps,
                                           const unsigned int lightCount)
{
    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int pageId = globalId / GI_PAGE_SIZE;
    unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
    unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;
    unsigned int segmentLocalVoxId = pageLocalId % GI_SEGMENT_SIZE;

    // Skip Whole segment if necessary
    if(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::EMPTY) return;
    assert(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) != SegmentOccupation::MARKED_FOR_CLEAR);

    // Fetch voxel
    CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
    if(voxelPosPacked == 0xFFFFFFFF) return;

    // Local Voxel pos and expand it if its one of the inner cascades
    uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
    uint3 voxelPos = ExpandToSVODepth(voxelUnpacked,
                                      cascadeNo,
                                      svoConstants.numCascades,
                                      svoConstants.totalDepth);


    unsigned int nodeIndex = 0;
    unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
    for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
    {
        CSVONode currentNode;
        if(i == svoConstants.denseDepth)
        {
            uint3 denseIndex = CalculateLevelVoxId(voxelPos, svoConstants.denseDepth,
                                                   svoConstants.totalDepth);

            assert(denseIndex.x < svoConstants.denseDim &&
                   denseIndex.y < svoConstants.denseDim &&
                   denseIndex.z < svoConstants.denseDim);

            currentNode = tex3D<unsigned int>(tSVODense,
                                              denseIndex.x,
                                              denseIndex.y,
                                              denseIndex.z);
        }
        else
        {
            currentNode = gSVOSparse[gLevelOffsets[i - svoConstants.denseDepth] + nodeIndex];
        }

        // Offset according to children
        assert(currentNode != 0xFFFFFFFF);
        unsigned int childIndex = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
        nodeIndex = currentNode + childIndex;
    }

    // Finally found location
    // Average color and normal
    // Fetch obj Id to get color
    ushort2 objectId;
    SegmentObjData objData = gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId];
    objectId.x = objData.objId;
    objectId.y = objData.batchId;
    unsigned int cacheVoxelId = objData.voxStride + segmentLocalVoxId;

    CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxNorm[pageLocalId];
    CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][cacheVoxelId].color);

    // Light Injection
    if(inject)
    {
        float4 colorSVO = UnpackSVOColor(voxelColorPacked);
        float4 normalSVO = UnpackSVONormal(voxelNormPacked);

        float3 worldPos =
        {
            outerCascadePos.x + voxelPos.x * span,
            outerCascadePos.y + voxelPos.y * span,
            outerCascadePos.z + voxelPos.z * span
        };

        // First Averager find and inject light
        float3 illum = LightInject(worldPos,

                                   colorSVO,
                                   normalSVO,

                                   camPos,
                                   camDir,

                                   lightVP,
                                   lightStruct,

                                   depthNear,
                                   depthFar,

                                   shadowMaps,
                                   lightCount,
                                   ambientColor);

        colorSVO.x = illum.x;
        colorSVO.y = illum.y;
        colorSVO.z = illum.z;
        voxelColorPacked = PackSVOColor(colorSVO);
    }

    // Atomic Average
	AtomicAvg(gSVOMat + matSparseOffset +
			  gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] +
			  nodeIndex,
			  voxelColorPacked,
			  voxelNormPacked,
			  1.0f);

    //gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 - 
    //		svoConstants.denseDepth] +
    //		nodeIndex] = PackSVOMaterial(voxelColorPacked, voxelNormPacked);
}

__global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,
										  cudaSurfaceObject_t sDenseMat,

										  const CSVONode* gSVODense,
										  const CSVONode* gSVOSparse,

										  const unsigned int* gLevelOffsets,
										  const unsigned int& gSVOLevelOffset,
										  const unsigned int& gSVONextLevelOffset,

										  const unsigned int levelNodeCount,
										  const unsigned int matOffset,
										  const unsigned int currentLevel,
										  const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int nodeId = globalId / 2;

	// Cull if out of range
	if(nodeId > levelNodeCount) return;

	// Read Sibling Materials
	const CSVONode* n = (currentLevel == svoConstants.denseDepth) ? gSVODense : gSVOSparse;
	CSVONode node = n[gSVOLevelOffset + nodeId];

	// Cull if there is no node no need to average
	if(node == 0xFFFFFFFF) return;

	// Only fetch parent when there a potential to have one
	bool fetchParentMat = ((svoConstants.totalDepth - currentLevel) < svoConstants.numCascades);

	uint64_t parentMat;
	if(globalId % 2 == 0) parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + nodeId].normalColor : 0x0;
	else parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + nodeId].anisoOccupancy : 0x0;

	// Average Portion
	// Material Data
	unsigned int count = 0;
	float4 avgSegment1 = {0.0f, 0.0f, 0.0f, 0.0f};
	float3 avgSegment2 = {0.0f, 0.0f, 0.0f};

	// Parent Incorporate
	if(parentMat != 0x0)
	{
		if(globalId % 2 == 0)
		{
			CSVOColor colorPacked;
			CVoxelNorm normalPacked;
			UnpackSVOMaterialColorNormal(colorPacked, normalPacked, parentMat);

			float4 color = UnpackSVOColor(colorPacked);
			float4 normal = UnpackSVONormal(normalPacked);

			avgSegment1.x = 8 * color.x;
			avgSegment1.y = 8 * color.y;
			avgSegment1.z = 8 * color.z;
			avgSegment1.w = 8 * color.w;

			avgSegment2.x = 8 * normal.x;
			avgSegment2.y = 8 * normal.y;
			avgSegment2.z = 8 * normal.z;
		}
		else
		{
			CSVOAnisoXY anisoXYPacked;
			CSVOAnisoZ anisoZPacked;
			UnpackSVOAnisoOccupancy(anisoXYPacked, anisoZPacked, parentMat);

			float4 anisoXY = UnpackSVOAnisoXY(anisoXYPacked);
			float4 anisoZ = UnpackSVOAnisoZ(anisoZPacked);

			avgSegment1.x = 8 * anisoXY.x;
			avgSegment1.y = 8 * anisoXY.y;
			avgSegment1.z = 8 * anisoXY.z;
			avgSegment1.w = 8 * anisoXY.w;
			avgSegment2.x = 8 * anisoZ.x;
			avgSegment2.y = 8 * anisoZ.y;
		}
		count += 8;
	}

	// Average
	if(node != 0xFFFFFFFF)
	{
		#pragma unroll
		for(unsigned int i = 0; i < 8; i++)
		{
			unsigned int currentNodeId = node + i;
			if(globalId % 2 == 0)
			{
				uint64_t mat = gSVOMat[matOffset + gSVONextLevelOffset + currentNodeId].normalColor;
				if(mat == 0x0) continue;
				CSVOColor colorPacked;
				CVoxelNorm normalPacked;
				UnpackSVOMaterialColorNormal(colorPacked, normalPacked, mat);
				float4 color = UnpackSVOColor(colorPacked);
				float4 normal = UnpackSVONormal(normalPacked);

				avgSegment1.x += color.x;
				avgSegment1.y += color.y;
				avgSegment1.z += color.z;
				avgSegment1.w += color.w;

				avgSegment2.x += normal.x;
				avgSegment2.y += normal.y;
				avgSegment2.z += normal.z;
			}
			else
			{
				uint64_t mat = gSVOMat[matOffset + gSVONextLevelOffset + currentNodeId].anisoOccupancy;
				if(mat == 0x0) continue;
				CSVOAnisoXY anisoXYPacked;
				CSVOAnisoZ anisoZPacked;
				UnpackSVOAnisoOccupancy(anisoXYPacked, anisoZPacked, mat);
				float4 anisoXY = UnpackSVOAnisoXY(anisoXYPacked);
				float4 anisoZ = UnpackSVOAnisoZ(anisoZPacked);

				/*if(i % 2 == 1)*/ avgSegment1.x += anisoXY.x;
				/*if(i % 2 == 0)*/ avgSegment1.y += anisoXY.y;
				/*if((i / 2) % 2 == 1)*/ avgSegment1.z += anisoXY.z;
				/*if((i / 2) % 2 == 0)*/ avgSegment1.w += anisoXY.w;
				/*if(i < 4)*/ avgSegment2.x += anisoZ.x;
				/*if(i > 4)*/ avgSegment2.y += anisoZ.y;
			}
			count++;
		}
	}

	// Divide by Count
	if(count == 0) count = 1.0f;
	float countInv = 1.0f / static_cast<float>(count);
	//float anisoInv = (count > 8) ? 0.125f : 0.25f;
	float anisoInv = (count > 8) ? 0.0625f : 0.125f;
	countInv = (globalId % 2 == 0) ? countInv : anisoInv;
	avgSegment1.x *= countInv;
	avgSegment1.y *= countInv;
	avgSegment1.z *= countInv;
	avgSegment1.w *= countInv;

	avgSegment2.x *= countInv;
	avgSegment2.y *= countInv;
	avgSegment2.z *= countInv;

	// Pack and Store	
	uint64_t averageValue;
	if(globalId % 2 == 0)
	{
		CSVOColor colorPacked = PackSVOColor({avgSegment1.x, avgSegment1.y, avgSegment1.z, avgSegment1.w});
		CVoxelNorm normPacked = PackSVONormal({avgSegment2.x, avgSegment2.y, avgSegment2.z, 0.0f});
		averageValue = PackSVOMaterialColorNormal(colorPacked, normPacked);
	}
	else
	{
		avgSegment1.x = fminf(avgSegment1.x, 1.0f);
		avgSegment1.y = fminf(avgSegment1.y, 1.0f);
		avgSegment1.z = fminf(avgSegment1.z, 1.0f);
		avgSegment1.w = fminf(avgSegment1.w, 1.0f);
		avgSegment2.x = fminf(avgSegment2.x, 1.0f);
		avgSegment2.y = fminf(avgSegment2.y, 1.0f);
		
		CSVOAnisoXY anisoXYPacked = PackAnisoXY(avgSegment1);
		CSVOAnisoZ anisoZPacked = PackAnisoZ({avgSegment2.x, avgSegment2.y, 0.0f, 0.0f});
		averageValue = PackSVOAnisoOccupancy(anisoXYPacked, anisoZPacked);
	}
	
    if(currentLevel == svoConstants.denseDepth)
    {
        int3 dim =
        {
            static_cast<int>(nodeId % svoConstants.denseDim),
            static_cast<int>((nodeId / svoConstants.denseDim) % svoConstants.denseDim),
            static_cast<int>(nodeId / (svoConstants.denseDim * svoConstants.denseDim))
        };
        uint2 data =
        {
            static_cast<unsigned int>(averageValue & 0x00000000FFFFFFFF),
            static_cast<unsigned int>(averageValue >> 32)
        };
		int dimX = (globalId % 2 == 0) ? (dim.x * sizeof(uint4)) : (dim.x * sizeof(uint4) + sizeof(uint2));
        surf3Dwrite(data, sDenseMat, dimX, dim.y, dim.z);
    }
    else
    {
		if(globalId % 2 == 0) gSVOMat[matOffset + gSVOLevelOffset + nodeId].normalColor = averageValue;
		else gSVOMat[matOffset + gSVOLevelOffset + nodeId].anisoOccupancy = averageValue;
    }
}

__global__ void SVOReconstructAverageNode(cudaSurfaceObject_t sDenseMatChild,
                                          cudaSurfaceObject_t sDenseMatParent,

                                          const unsigned int parentSize)
{
    // Linear Id
    unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int parentId = globalId / GI_DENSE_WORKER_PER_PARENT;

    // 3D Id
    char3 idMap = voxLookup[globalId % GI_DENSE_WORKER_PER_PARENT];
    uint3 parentId3D =
    {
        static_cast<unsigned int>(parentId % parentSize),
        static_cast<unsigned int>((parentId / parentSize) % parentSize),
        static_cast<unsigned int>(parentId / (parentSize * parentSize))
    };
    uint3 childId3D =
    {
        parentId3D.x * 2 + idMap.x,
        parentId3D.y * 2 + idMap.y,
        parentId3D.z * 2 + idMap.z
    };

    // 3D Fetch
    uint4 data;
    surf3Dread(&data, sDenseMatChild,
               childId3D.x * sizeof(uint4),
               childId3D.y,
               childId3D.z);

    // Data
    unsigned int count = (data.x == 0 && data.y == 0 && data.z == 0 && data.w == 0) ? 0 : 1;
    float4 color = UnpackSVOColor(data.x);
    float4 normal = UnpackSVONormal(data.y);
	float4 anisoXY = UnpackSVOAnisoXY(data.z);
	float4 anisoZ = UnpackSVOAnisoXY(data.w);

    // Average	
    #pragma unroll
    for(int offset = GI_DENSE_WORKER_PER_PARENT / 2; offset > 0; offset /= 2)
    {
        color.x += __shfl_down(color.x, offset, GI_DENSE_WORKER_PER_PARENT);
        color.y += __shfl_down(color.y, offset, GI_DENSE_WORKER_PER_PARENT);
        color.z += __shfl_down(color.z, offset, GI_DENSE_WORKER_PER_PARENT);
        color.w += __shfl_down(color.w, offset, GI_DENSE_WORKER_PER_PARENT);

        normal.x += __shfl_down(normal.x, offset, GI_DENSE_WORKER_PER_PARENT);
        normal.y += __shfl_down(normal.y, offset, GI_DENSE_WORKER_PER_PARENT);
        normal.z += __shfl_down(normal.z, offset, GI_DENSE_WORKER_PER_PARENT);
        //normal.w += __shfl_down(normal.w, offset, GI_DENSE_WORKER_PER_PARENT);

		unsigned int i = globalId % GI_DENSE_WORKER_PER_PARENT;
		anisoXY.x += __shfl_down((i % 2 == 1) ? anisoXY.x : 0.0f, offset, GI_DENSE_WORKER_PER_PARENT);
		anisoXY.y += __shfl_down((i % 2 == 0) ? anisoXY.y : 0.0f, offset, GI_DENSE_WORKER_PER_PARENT);
		anisoXY.z += __shfl_down(((i / 2) % 2 == 1) ? anisoXY.z : 0.0f, offset, GI_DENSE_WORKER_PER_PARENT);
		anisoXY.w += __shfl_down(((i / 2) % 2 == 0) ? anisoXY.w : 0.0f, offset, GI_DENSE_WORKER_PER_PARENT);

		anisoZ.x += __shfl_down((i < 4) ? anisoXY.w : 0.0f, offset, GI_DENSE_WORKER_PER_PARENT);
		anisoZ.y += __shfl_down((i > 4) ? anisoXY.w : 0.0f, offset, GI_DENSE_WORKER_PER_PARENT);

        count += __shfl_down(count, offset, GI_DENSE_WORKER_PER_PARENT);
    }

    // Division
    float countInv = 1.0f / ((count != 0) ? float(count) : 1.0f);
    color.x *= countInv;
    color.y *= countInv;
    color.z *= countInv;
    color.w *= countInv;

    normal.x *= countInv;
    normal.y *= countInv;
    normal.z *= countInv;

	anisoXY.x *= 0.25f;
	anisoXY.y *= 0.25f;
	anisoXY.z *= 0.25f;
	anisoXY.w *= 0.25f;

	anisoZ.x *= 0.25f;
	anisoZ.y *= 0.25f;

    data.x = PackSVOColor(color);
    data.y = PackSVONormal(normal);
	data.z = PackAnisoXY(anisoXY);
	data.w = PackAnisoZ(anisoZ);

    if(globalId % GI_DENSE_WORKER_PER_PARENT == 0 && count != 0)
    {
        surf3Dwrite(data, sDenseMatParent,
                    parentId3D.x * sizeof(uint4),
                    parentId3D.y,
                    parentId3D.z);
    }
}

__global__ void SVOReconstruct(CSVOMaterial* gSVOMat,
							   CSVONode* gSVOSparse,
							   CSVONode* gSVODense,
							   unsigned int* gLevelAllocators,

							   const unsigned int* gLevelOffsets,
							   const unsigned int* gLevelTotalSizes,

							   // For Color Lookup
							   const CVoxelPage* gVoxelData,
							   CVoxelColor** gVoxelRenderData,

							   const unsigned int matSparseOffset,
							   const unsigned int cascadeNo,
							   const CSVOConstants& svoConstants,

							   // Light Inject Related
							   bool inject,
							   float span,
							   const float3 outerCascadePos,
							   const float3 ambientColor,

							   const float4 camPos,
							   const float3 camDir,

							   const CMatrix4x4* lightVP,
							   const CLight* lightStruct,

							   const float depthNear,
							   const float depthFar,

							   cudaTextureObject_t shadowMaps,
							   const unsigned int lightCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;
	unsigned int segmentLocalVoxId = pageLocalId % GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::EMPTY) return;
	assert(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) != SegmentOccupation::MARKED_FOR_CLEAR);

	// Fetch voxel
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
									  svoConstants.numCascades,
									  svoConstants.totalDepth);

	// ObjId Fetch
	ushort2 objectId;
	SegmentObjData objData = gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId];
	objectId.x = objData.objId;
	objectId.y = objData.batchId;
	unsigned int cacheVoxelId = objData.voxStride + segmentLocalVoxId;

	CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxNorm[pageLocalId];
	CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][cacheVoxelId].color);
	CVoxelOccupancy voxOccupPacked = gVoxelData[pageId].dGridVoxOccupancy[pageLocalId];

	// Unpack Occupancy
	uint3 neigbourBits;
	float3 weights;
	ExpandOccupancy(neigbourBits, weights, voxOccupPacked);
	int3 voxOffset = 
	{
		static_cast<int>(neigbourBits.x),
		static_cast<int>(neigbourBits.y),
		static_cast<int>(neigbourBits.z)
	};
	voxOffset.x = 2 * voxOffset.x - 1;
	voxOffset.y = 2 * voxOffset.y - 1;
	voxOffset.z = 2 * voxOffset.z - 1;

	// Light Injection
	if(inject)
	{
		float4 colorSVO = UnpackSVOColor(voxelColorPacked);
		float4 normalSVO = UnpackSVONormal(voxelNormPacked);

		float3 worldPos =
		{
			outerCascadePos.x + voxelPos.x * span,
			outerCascadePos.y + voxelPos.y * span,
			outerCascadePos.z + voxelPos.z * span
		};

		// First Averager find and inject light
		float3 illum = LightInject(worldPos,

								   colorSVO,
								   normalSVO,

								   camPos,
								   camDir,

								   lightVP,
								   lightStruct,

								   depthNear,
								   depthFar,

								   shadowMaps,
								   lightCount,
								   ambientColor);

		colorSVO.x = illum.x;
		colorSVO.y = illum.y;
		colorSVO.z = illum.z;
		voxelColorPacked = PackSVOColor(colorSVO);
	}


	//printf("cascadeNo %d weights %f, %f, %f\n", cascadeNo, weights.x, weights.y, weights.z);

	float totalOccupancy = 0.0f;
	for(unsigned int i = 0; i < GI_SVO_WORKER_PER_NODE; i++)
	{
		// Create NeigNode
		uint3 currentVoxPos = voxelPos;
		unsigned int cascadeOffset = svoConstants.numCascades - cascadeNo - 1;
		currentVoxPos.x += voxLookup[i].x * (voxOffset.x << cascadeOffset);
		currentVoxPos.y += voxLookup[i].y * (voxOffset.y << cascadeOffset);
		currentVoxPos.z += voxLookup[i].z * (voxOffset.z << cascadeOffset);
		


		// Calculte this nodes occupancy
		float occupancy = 1.0f;
		float3 volume;
		volume.x = (voxLookup[i].x == 1) ? weights.x : (1.0f - weights.x);
		volume.y = (voxLookup[i].y == 1) ? weights.y : (1.0f - weights.y);
		volume.z = (voxLookup[i].z == 1) ? weights.z : (1.0f - weights.z);
		occupancy = volume.x * volume.y * volume.z;
		totalOccupancy += occupancy;

		//printf("(%d, %d, %d) occupancy %f\n",
		//	   voxLookup[i].z, voxLookup[i].y, voxLookup[i].x,
		//	   occupancy);

		unsigned int location;
		unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
		for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
		{
			unsigned int levelIndex = i - svoConstants.denseDepth;
			CSVONode* node = nullptr;
			if(i == svoConstants.denseDepth)
			{
				uint3 levelVoxId = CalculateLevelVoxId(currentVoxPos, i, svoConstants.totalDepth);
				node = gSVODense +
					svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
					svoConstants.denseDim * levelVoxId.y +
					levelVoxId.x;
			}
			else
			{
				node = gSVOSparse + gLevelOffsets[levelIndex] + location;
			}

			// Allocate (or acquire) next location
			location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1]);
			assert(location < gLevelTotalSizes[levelIndex + 1]);

			// Offset child
			unsigned int childId = CalculateLevelChildId(currentVoxPos, i + 1, svoConstants.totalDepth);
			location += childId;
		}

		AtomicAvg(gSVOMat + matSparseOffset +
				  gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] + location,
				  voxelColorPacked,
				  voxelNormPacked,
				  occupancy);
	}

	//printf("total occupancy %f\n", totalOccupancy);
	//// Non atmoic overwrite
	//gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 -
	//		svoConstants.denseDepth] +
	//		nodeIndex] = PackSVOMaterial(voxelColorPacked, voxelNormPacked);
}

//
//__global__ void SVOReconstruct(CSVOMaterial* gSVOMat,
//							   CSVONode* gSVOSparse,
//							   CSVONode* gSVODense,
//							   unsigned int* gLevelAllocators,
//
//							   const unsigned int* gLevelOffsets,
//							   const unsigned int* gLevelTotalSizes,
//
//							   // For Color Lookup
//							   const CVoxelPage* gVoxelData,
//							   CVoxelColor** gVoxelRenderData,
//
//							   const unsigned int matSparseOffset,
//							   const unsigned int cascadeNo,
//							   const CSVOConstants& svoConstants,
//
//							   // Light Inject Related
//							   bool inject,
//							   float span,
//							   const float3 outerCascadePos,
//							   const float3 ambientColor,
//
//							   const float4 camPos,
//							   const float3 camDir,
//
//							   const CMatrix4x4* lightVP,
//							   const CLight* lightStruct,
//
//							   const float depthNear,
//							   const float depthFar,
//
//							   cudaTextureObject_t shadowMaps,
//							   const unsigned int lightCount)
//{
//	const unsigned int NODE_PER_BLOCK = GI_THREAD_PER_BLOCK / GI_SVO_WORKER_PER_NODE;
//
//	__shared__ CVoxelPos voxPos[NODE_PER_BLOCK];
//	__shared__ CVoxelNorm voxNorm[NODE_PER_BLOCK];
//	__shared__ CSVOColor voxColor[NODE_PER_BLOCK];
//	__shared__ CVoxelOccupancy voxOccupancy[NODE_PER_BLOCK];
//	__shared__ bool abortSignal;
//
//	// Linear Local Ids (Per Block Node)
//	unsigned int mNodeId = threadIdx.x + blockIdx.x * NODE_PER_BLOCK;
//	unsigned int mPageId = mNodeId / GI_PAGE_SIZE;
//	unsigned int mPageLocalId = mNodeId % GI_PAGE_SIZE;
//	unsigned int mPageLocalSegmentId = mPageLocalId / GI_SEGMENT_SIZE;
//	unsigned int mSegmentLocalVoxId = mPageLocalId % GI_SEGMENT_SIZE;
//
//	unsigned int blockLocalId = threadIdx.x;
//	if(blockLocalId < NODE_PER_BLOCK)
//	{
//		// Skip Whole segment if necessary
//		if(mNodeId == 0)
//		{
//			assert(ExpandOnlyOccupation(gVoxelData[mPageId].dSegmentObjData[mPageLocalSegmentId].packed) != SegmentOccupation::MARKED_FOR_CLEAR);
//			abortSignal = ExpandOnlyOccupation(gVoxelData[mPageId].dSegmentObjData[mPageLocalSegmentId].packed) == SegmentOccupation::EMPTY;
//		}
//	}
//	__syncthreads();
//	if(abortSignal) return;
//
//	if(blockLocalId < NODE_PER_BLOCK)
//	{
//		CVoxelPos voxelPosPacked = gVoxelData[mPageId].dGridVoxPos[mPageLocalId];
//		CVoxelNorm voxelNormPacked = 0xFFFFFFFF;
//		CSVOColor voxelColorPacked = 0xFFFFFFFF;
//		CVoxelOccupancy voxelOccupancyPacked = 0xFFFFFFFF;
//		if(voxelPosPacked != 0xFFFFFFFF)
//		{
//			// ObjId Fetch
//			ushort2 objectId;
//			SegmentObjData objData = gVoxelData[mPageId].dSegmentObjData[mPageLocalSegmentId];
//			objectId.x = objData.objId;
//			objectId.y = objData.batchId;
//			unsigned int cacheVoxelId = objData.voxStride + mSegmentLocalVoxId;
//
//			voxelNormPacked = gVoxelData[mPageId].dGridVoxNorm[mPageLocalId];
//			//voxelColorPacked = *reinterpret_cast<CSVOColor*>(&gVoxelRenderData[objectId.y][cacheVoxelId].color);
//			voxelOccupancyPacked = gVoxelData[mPageId].dGridVoxOccupancy[mPageLocalId];
//
//			//// Light Injection
//			//if(inject)
//			//{
//			//	float4 colorSVO = UnpackSVOColor(voxelColorPacked);
//			//	float4 normalSVO = UnpackSVONormal(voxelNormPacked);
//			//	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
//			//	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
//			//									  svoConstants.numCascades,
//			//									  svoConstants.totalDepth);
//
//			//	float3 worldPos =
//			//	{
//			//		outerCascadePos.x + voxelPos.x * span,
//			//		outerCascadePos.y + voxelPos.y * span,
//			//		outerCascadePos.z + voxelPos.z * span
//			//	};
//
//			//	// First Averager find and inject light
//			//	float3 illum = LightInject(worldPos, colorSVO, normalSVO,
//			//							   camPos, camDir,
//			//							   lightVP, lightStruct,
//			//							   depthNear, depthFar,
//			//							   shadowMaps, lightCount, ambientColor);
//
//			//	colorSVO.x = illum.x;
//			//	colorSVO.y = illum.y;
//			//	colorSVO.z = illum.z;
//			//	voxelColorPacked = PackSVOColor(colorSVO);
//			//}
//		}
//		// Write back
//		voxPos[blockLocalId] = voxelPosPacked;
//		voxNorm[blockLocalId] = voxelNormPacked;
//		//voxColor[blockLocalId] = voxelColorPacked;
//		voxOccupancy[blockLocalId] = voxelOccupancyPacked;
//	}
//	__syncthreads();
//
//	//if(blockLocalId >= NODE_PER_BLOCK) return;
//
//	// Now Fetch
//	//unsigned int localNodeId = blockLocalId % NODE_PER_BLOCK;
//	unsigned int localNodeId = blockLocalId / GI_SVO_WORKER_PER_NODE;
//	CVoxelPos voxelPosPacked = voxPos[localNodeId];
//	CVoxelNorm voxelNormPacked = voxNorm[localNodeId];
//	CSVOColor voxelColorPacked = 0; voxColor[localNodeId];
//	CVoxelOccupancy voxelOccupancyPacked = voxOccupancy[localNodeId];
//
//	//CVoxelPos voxelPosPacked = gVoxelData[mPageId].dGridVoxPos[mPageLocalId];
//	//CVoxelNorm voxelNormPacked = gVoxelData[mPageId].dGridVoxNorm[mPageLocalId];
//	//CSVOColor voxelColorPacked = 0;// *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][cacheVoxelId].color);
//	//CVoxelOccupancy voxelOccupancyPacked = gVoxelData[mPageId].dGridVoxOccupancy[mPageLocalId];
//
//	// Stop if this sub segment is useless
//	if(voxelPosPacked == 0xFFFFFFFF) return;	
//
////	if(blockLocalId % GI_SVO_WORKER_PER_NODE != 0) return;
//
//	// Local Voxel pos and expand it if its one of the inner cascades
//	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
//	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
//									  svoConstants.numCascades,
//									  svoConstants.totalDepth);
//
//	unsigned int location;
//	unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
//	for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
//	{
//		unsigned int levelIndex = i - svoConstants.denseDepth;
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
//		unsigned int childId = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
//		location += childId;
//	}
//
//	AtomicAvg(gSVOMat + matSparseOffset +
//			  gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] + location,
//			  voxelColorPacked,
//			  voxelNormPacked,
//			  1.0f);
//
//	//// Non atmoic overwrite
//	//gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 -
//	//		svoConstants.denseDepth] +
//	//		nodeIndex] = PackSVOMaterial(voxelColorPacked, voxelNormPacked);
//}
