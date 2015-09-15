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
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = (globalId - pageId * GI_PAGE_SIZE);

	if(gVoxPages[pageId].dGridVoxNormPos[pageLocalId].y != 0)
	   atomicAdd(&totalVox, 1);
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
	if(gVoxPages[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxPages[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Data Read
	CVoxelNormPos voxelNormalPos = gVoxPages[pageId].dGridVoxNormPos[pageLocalId];
	CVoxelIds voxelIds = gVoxPages[pageId].dGridVoxIds[pageLocalId];

	// Cull Check
	ushort2 objectId;
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, voxelIds);
	
	// Zero normal means invalid voxel
	if(voxelNormalPos.y != 0)
	{
		unsigned int index = atomicInc(&atomicIndex, 0xFFFFFFFF);
		
		voxelData[index] = CVoxelPacked {voxelNormalPos.x, voxelNormalPos.y, voxelIds.x, voxelIds.y};
		voxelColorData[index] = gVoxelRenderData[objectId.y][voxelId].color;
	}
}