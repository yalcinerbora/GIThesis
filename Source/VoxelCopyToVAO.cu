#include "VoxelCopyToVAO.cuh"
#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "CVoxelPage.h"
#include <cstdio>
#include <cassert>

__global__ void DetermineTotalVoxCount(int& totalVox,

									   const CVoxelPage* gVoxPages,
									   const CVoxelGrid& gGridInfo,
									   const uint32_t pageCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= pageCount * GI_SEGMENT_PER_PAGE) return;

	if(gVoxPages[globalId / GI_SEGMENT_PER_PAGE].dIsSegmentOccupied[globalId % GI_SEGMENT_PER_PAGE] == 1)
		atomicAdd(&totalVox, GI_SEGMENT_SIZE);
}

__global__ void DetermineTotalVoxCount(int& totalVox,

									   const ushort2* gObjectAllocLocations,
									   const uint32_t segmentCount,

									   const unsigned int* gObjectAllocIndexLookup,
									   const CObjectVoxelInfo* gVoxelInfo,
									   const CObjectTransform* gObjTransforms,
									   const uint32_t objCount,
									   
									   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objCount) return;


	float3 scaling = ExtractScaleInfo(gObjTransforms[globalId].transform);
	assert(scaling.x == scaling.y);
	assert(scaling.y == scaling.z);
	unsigned int voxelDim = static_cast<unsigned int>(gVoxelInfo[globalId].span * scaling.x / gGridInfo.span);
	unsigned int voxScale = voxelDim == 0 ? 0 : 1;
	unsigned int voxelCount = (((gVoxelInfo[globalId].voxelCount * voxScale) + GI_SEGMENT_SIZE - 1) / GI_SEGMENT_SIZE) * GI_SEGMENT_SIZE;

	unsigned int objSegmentLoc = gObjectAllocIndexLookup[globalId];
	if(gObjectAllocLocations[objSegmentLoc].x != 0xFFFF)
	{
		atomicAdd(&totalVox, voxelCount);
	}
}

__global__ void VoxelCopyToVAO(// Two ogl Buffers for rendering used voxels
							   CVoxelPacked* voxelData,
							   uchar4* voxelColorData,
							   unsigned int& atomicIndex,

							   // Per Obj Segment
							   ushort2** gObjectAllocLocations,

							   // Per obj
							   unsigned int** gObjectAllocIndexLookup,

							   // Per vox
							   CVoxelRender** gVoxelRenderData,

							   // Page
							   const CVoxelPage* gVoxPages,
							   uint32_t pageCount,
							   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = blockIdx.x / GI_BLOCK_PER_PAGE;
	unsigned int pageLocalId = globalId - (pageId * GI_PAGE_SIZE);
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;
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