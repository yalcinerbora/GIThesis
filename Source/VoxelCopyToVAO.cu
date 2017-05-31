//#include "VoxelCopyToVAO.cuh"
//#include "COpenGLTypes.h"
//#include "CVoxelFunctions.cuh"
//#include "CSVOTypes.h"
//#include <cstdio>
//#include <cassert>
//#include "GIVoxelPages.h"
//
//__global__ void VoxCountPage(int& totalVox,
//
//							 const CVoxelPage* gVoxPages,
//							 const CVoxelGrid& gGridInfo,
//							 const uint32_t pageCount)
//{
//	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int pageId = globalId / GIVoxelPages::PageSize;
//	unsigned int pageLocalId = (globalId - pageId * GIVoxelPages::PageSize);
//
//	// All one normal means invalid voxel
//	if(gVoxPages[pageId].dGridVoxPos[pageLocalId] != 0xFFFFFFFF)
//	   atomicAdd(&totalVox, 1);
//}
//
//__global__ void VoxCpyPage(// Two ogl Buffers for rendering used voxels
//						   CVoxelNormPos* voxelNormPosData,
//						   uchar4* voxelColorData,
//						   unsigned int& atomicIndex,
//						   const unsigned int maxBufferSize,
//
//						   // Per Obj Segment
//						   ushort2** gObjectAllocLocations,
//
//						   // Per obj
//						   unsigned int** gObjectAllocIndexLookup,
//
//						   // Per vox
//						   CVoxelAlbedo** gVoxelRenderData,
//
//						   // Page
//						   const CVoxelPage* gVoxPages,
//						   uint32_t pageCount,
//						   const CVoxelGrid& gGridInfo)
//{
//	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
//	unsigned int pageId = blockIdx.x / GIVoxelPages::BlockPerPage;
//	unsigned int pageLocalId = globalId - (pageId * GIVoxelPages::PageSize);
//	unsigned int pageLocalSegmentId = pageLocalId / GIVoxelPages::SegmentSize;
//	unsigned int segmentLocalVoxId = pageLocalId % GIVoxelPages::SegmentSize;
//
//	// Skip whole segment if necessary
//	if(ExpandOnlyOccupation(gVoxPages[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::EMPTY) return;
//	assert(ExpandOnlyOccupation(gVoxPages[pageId].dSegmentObjData[pageLocalSegmentId].packed) != SegmentOccupation::MARKED_FOR_CLEAR);
//
//	// Data Read
//	CVoxelPos voxPosPacked = gVoxPages[pageId].dGridVoxPos[pageLocalId];
//	
//	// All one normal means invalid voxel
//	if(voxPosPacked != 0xFFFFFFFF)
//	{	
//		CVoxelPos voxNormpacked = gVoxPages[pageId].dGridVoxNorm[pageLocalId];
//
//		unsigned int index = atomicInc(&atomicIndex, 0xFFFFFFFF);
//		assert(index < maxBufferSize);
//
//		// Fetch obj Id to get color
//		// ObjId Fetch
//		ushort2 objectId;
//		SegmentObjData objData = gVoxPages[pageId].dSegmentObjData[pageLocalSegmentId];
//		objectId.x = objData.objId;
//		objectId.y = objData.batchId;
//		unsigned int cacheVoxelId = objData.voxStride + segmentLocalVoxId;
//
//		voxelNormPosData[index] = uint2{voxPosPacked, voxNormpacked};
//		voxelColorData[index] = gVoxelRenderData[objectId.y][cacheVoxelId];
//	}
//}
//
