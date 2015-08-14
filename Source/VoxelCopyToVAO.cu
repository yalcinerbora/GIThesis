#include "VoxelCopyToVAO.cuh"
#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "CVoxelPage.h"
#include <cstdio>

__global__ void DetermineTotalVoxCount(int& totalVox,

									   const CVoxelPage* gVoxPages,
									   const CVoxelGrid& gGridInfo,
									   const uint32_t pageCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= pageCount * GI_SEGMENT_PER_PAGE) return;

	if(gVoxPages[globalId / GI_SEGMENT_PER_PAGE].dIsSegmentOccupied[globalId % GI_SEGMENT_PER_PAGE] == 1)
		atomicAdd(&totalVox, GI_SEGMENT_SIZE);

	//float3 scaling = ExtractScaleInfo(gObjTransforms[globalId].transform);
	//uint3 voxelDim;
	//voxelDim.x = static_cast<unsigned int>(gVoxelInfo[globalId].span * scaling.x / gGridInfo.span);
	//voxelDim.y = static_cast<unsigned int>(gVoxelInfo[globalId].span * scaling.y / gGridInfo.span);
	//voxelDim.z = static_cast<unsigned int>(gVoxelInfo[globalId].span * scaling.z / gGridInfo.span);
	//unsigned int voxScale = voxelDim.x * voxelDim.y * voxelDim.z;
	//unsigned int voxelCount = (((gVoxelInfo[globalId].voxelCount * voxScale) + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE) * GI_SEGMENT_SIZE;

	//unsigned int objSegmentLoc = gObjectAllocIndexLookup[globalId];
	//if(objSegmentLoc < segmentCount &&
	//   gObjectAllocLocations[objSegmentLoc].x != 0xFFFF)
	//{
	//	atomicAdd(&totalVox, voxelCount);
	//	//printf("%d : %#06x, %#06x\n", globalId, gObjectAllocLocations[gObjectAllocIndexLookup[globalId]].x, gObjectAllocLocations[gObjectAllocIndexLookup[globalId]].y);
	//}
}

__global__ void VoxelCopyToVAO(// Two ogl Buffers for rendering used voxels
							   uint4* voxelData,
							   uchar4* voxelColorData,
							   unsigned int& atomicIndex,

							   // Per Obj Segment
							   const ushort2** gObjectAllocLocations,

							   // Per obj
							   const unsigned int** gObjectAllocIndexLookup,

							   // Per vox
							   const CVoxelRender** gVoxelRenderData,

							   // Data
							   const CVoxelPage* gVoxPages,
							   uint32_t pageCount,
							   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = blockIdx.x / GI_BLOCK_PER_PAGE;
	unsigned int pageLocalId = globalId - (blockIdx.x / GI_BLOCK_PER_PAGE);
	unsigned int pageLocalSegmentId = globalId - (blockIdx.x / GI_BLOCK_PER_PAGE) / GI_SEGMENT_SIZE;
	if(gVoxPages[pageId].dIsSegmentOccupied[pageLocalSegmentId] == 0) return;

	// Unpacking
	CVoxelPacked voxelPacked = gVoxPages[pageId].dGridVoxels[pageLocalId];
	ushort2 objectId;
	uint3 voxPos;
	float3 normal;
	unsigned int voxelSpanRatio;
	unsigned int renderLoc;
	ExpandVoxelData(voxPos, normal, objectId, renderLoc, voxelSpanRatio, voxelPacked);
	
	if(gObjectAllocLocations[objectId.x][gObjectAllocIndexLookup[objectId.x][objectId.y]].x != 0xFFFF)
	{
		unsigned int index = atomicInc(&atomicIndex, 0xFFFFFFFF);
		voxelData[index] = voxelPacked;
		voxelColorData[index] = gVoxelRenderData[objectId.x][renderLoc].color;
	}
}