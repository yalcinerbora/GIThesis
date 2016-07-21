#include "VoxelCopyToVAO.cuh"
#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"
#include "CSVOTypes.cuh"
#include "CVoxelPage.h"
#include <cstdio>
#include <cassert>

__global__ void VoxCountPage(int& totalVox,

							 const CVoxelPage* gVoxPages,
							 const CVoxelGrid& gGridInfo,
							 const uint32_t pageCount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = (globalId - pageId * GI_PAGE_SIZE);

	// All one normal means invalid voxel
	if(gVoxPages[pageId].dGridVoxPos[pageLocalId] != 0xFFFFFFFF)
	   atomicAdd(&totalVox, 1);
}

__global__ void VoxCpyPage(// Two ogl Buffers for rendering used voxels
						   CVoxelNormPos* voxelNormPosData,
						   uchar4* voxelColorData,
						   unsigned int& atomicIndex,
						   const unsigned int maxBufferSize,

						   // Per Obj Segment
						   ushort2** gObjectAllocLocations,

						   // Per obj
						   unsigned int** gObjectAllocIndexLookup,

						   // Per vox
						   CVoxelColor** gVoxelRenderData,

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
	CVoxelPos voxPosPacked = gVoxPages[pageId].dGridVoxPos[pageLocalId];
	
	// All one normal means invalid voxel
	if(voxPosPacked != 0xFFFFFFFF)
	{	
		CVoxelPos voxNormpacked = gVoxPages[pageId].dGridVoxNorm[pageLocalId];

		unsigned int index = atomicInc(&atomicIndex, 0xFFFFFFFF);
		assert(index < maxBufferSize);

		// Fetch obj Id to get color
		ushort2 objectId;
		CVoxelObjectType objType;
		unsigned int voxelId;
		ExpandVoxelIds(voxelId, objectId, objType, gVoxPages[pageId].dGridVoxIds[pageLocalId]);

		voxelNormPosData[index] = uint2{voxPosPacked, voxNormpacked};
		voxelColorData[index] = gVoxelRenderData[objectId.y][voxelId].color;
	}
}

