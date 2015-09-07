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

	// Data Read
	CVoxelNormPos voxelNormalPos = gVoxPages[pageId].dGridVoxNormPos[pageLocalId];
	CVoxelIds voxelIds = gVoxPages[pageId].dGridVoxIds[pageLocalId];

	// Cull Check
	ushort2 objectId;
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, voxelIds);
	
//	if(gObjectAllocLocations[objectId.y][gObjectAllocIndexLookup[objectId.y][objectId.x]].x != 0xFFFF)
	{
		unsigned int index = atomicInc(&atomicIndex, 0xFFFFFFFF);
		
		
		unsigned int value = 0;
		value |= static_cast<unsigned int>(16) << 27;
		value |= static_cast<unsigned int>(256) << 18;	// Z
		value |= static_cast<unsigned int>(256) << 9;	// Y
		value |= static_cast<unsigned int>(456);		// X
		voxelData[index] = CVoxelPacked { value, 0, 0, 0 };
		voxelColorData[index] = uchar4 { 255, 0, 255, 255 };


		//voxelData[index] = CVoxelPacked {voxelNormalPos.x, voxelNormalPos.y, voxelIds.x, voxelIds.y};
		//voxelColorData[index] = gVoxelRenderData[objectId.y][voxelId].color;

	}
}