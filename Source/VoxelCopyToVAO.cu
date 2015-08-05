#include "VoxelCopyToVAO.cuh"
#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "CVoxelPage.h"

__device__ void DetermineTotalVoxCount(int& totalVox,

									   // Per Obj Segment
									   const ushort2* gObjectAllocLocations,

									   // Per obj
									   const unsigned int* gObjectAllocIndexLookup,
									   const CObjectVoxelInfo* gObjInfo,
									   uint32_t objectCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= objectCount) return;

	if(gObjectAllocLocations[gObjectAllocIndexLookup[globalId]].x != 0xFFFF)
	{
		atomicAdd(&totalVox, gObjInfo[globalId].voxelCount);
	}
}

__device__ void VoxelCopyToVAO(// Two ogl Buffers for rendering used voxels
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
	unsigned int renderLoc;
	ExpandVoxelData(voxPos, normal, objectId, renderLoc, voxelPacked);
	
	if(gObjectAllocLocations[objectId.x][gObjectAllocIndexLookup[objectId.x][objectId.y]].x != 0xFFFF)
	{
		unsigned int index = atomicInc(&atomicIndex, 0xFFFFFFFF);
		voxelData[index] = voxelPacked;
		voxelColorData[index] = gVoxelRenderData[objectId.x][renderLoc].color;
	}
}