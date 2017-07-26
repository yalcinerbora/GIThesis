#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>

#include "PageKernels.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"
#include "COpenGLCommon.cuh"
#include "CAtomicAlloc.cuh"

__global__ void VoxelObjectDealloc(// Voxel System
								   CVoxelPage* gVoxelData,
								   const CVoxelGrid& gGridInfo,

								   // Per Object Segment Related
								   ushort2* gObjectAllocLocations,
								   const SegmentObjData* gSegmentObjectData,
								   const uint32_t totalSegments,

								   // Per Object Related
								   const CObjectAABB* gObjectAABB,
								   const CObjectTransform* gObjTransforms,
								   const unsigned int* gObjTransformIds)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Now Thread Scheme changes per objectSegment
	if(globalId >= totalSegments) return;
	
	// Determine Obj Id
	SegmentObjData segObjData = gSegmentObjectData[globalId];
	if(segObjData.objId == 0xFFFF) return;

	const uint32_t transformId = gObjTransformIds[segObjData.objId];
	const CMatrix4x4 transform = gObjTransforms[transformId].transform;
	CObjectAABB objAABB = gObjectAABB[segObjData.objId];
	bool intersects = CheckGridVoxIntersect(gGridInfo, objAABB, transform);

	// Check if this object is not allocated
	ushort2 objAlloc = gObjectAllocLocations[globalId];
	if(!intersects && objAlloc.x != 0xFFFF)
	{
		// "Dealocate"
		assert(ExpandOnlyOccupation(gVoxelData[objAlloc.x].dSegmentObjData[objAlloc.y].packed) == SegmentOccupation::OCCUPIED);
		unsigned int size = AtomicDealloc(&(gVoxelData[objAlloc.x].dEmptySegmentStackSize), GI_SEGMENT_PER_PAGE);
		assert(size != GI_SEGMENT_PER_PAGE);
		if(size != GI_SEGMENT_PER_PAGE)
		{
			unsigned int location = size;
			gVoxelData[objAlloc.x].dEmptySegmentPos[location] = objAlloc.y;

			SegmentObjData segObjId = {0};
			segObjId.packed = PackSegmentPacked(CVoxelObjectType::STATIC,
												SegmentOccupation::MARKED_FOR_CLEAR, 0);
			gVoxelData[objAlloc.x].dSegmentObjData[objAlloc.y] = segObjId;
			gObjectAllocLocations[globalId] = ushort2{0xFFFF, 0xFFFF};
		}
	}
}

__global__ void VoxelObjectAlloc(// Voxel System
								 CVoxelPage* gVoxelData,
								 const unsigned int gPageAmount,
								 const CVoxelGrid& gGridInfo,

								 // Per Object Segment Related
								 ushort2* gObjectAllocLocations,
								 const SegmentObjData* gSegmentObjectData,
								 const uint32_t totalSegments,

								 // Per Object Related
								 const CObjectAABB* gObjectAABB,
								 const CObjectTransform* gObjTransforms,
								 const unsigned int* gObjTransformIds)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= totalSegments) return;
	
	// Determine Obj Id (-1 Id means this object is too small for this grid)
	SegmentObjData segObjData = gSegmentObjectData[globalId];
	if(segObjData.objId == 0xFFFF) return;

	// Intersection Check
	const uint32_t transformId = gObjTransformIds[segObjData.objId];
	const CMatrix4x4 transform = gObjTransforms[transformId].transform;
	const CObjectAABB objAABB = gObjectAABB[segObjData.objId];
	bool intersects = CheckGridVoxIntersect(gGridInfo, objAABB, transform);

	// Check if this object already allocated
	ushort2 objAlloc = gObjectAllocLocations[globalId];
	if(intersects && objAlloc.x == 0xFFFF)
	{
		// "Allocate"
		// Check page by page
		for(unsigned int i = 0; i < gPageAmount; i++)
		{
			unsigned int size = AtomicAlloc(&(gVoxelData[i].dEmptySegmentStackSize));
			if(size != 0)
			{
				unsigned int location = gVoxelData[i].dEmptySegmentPos[size - 1];
				assert(ExpandOnlyOccupation(gVoxelData[i].dSegmentObjData[location].packed) == SegmentOccupation::EMPTY);
				gObjectAllocLocations[globalId] = ushort2
				{
					static_cast<unsigned short>(i),
					static_cast<unsigned short>(location)
				};
				gVoxelData[i].dSegmentObjData[location] = segObjData;
				return;
			}
		}
	}
}

__global__ void VoxelClearMarked(CVoxelPage* gVoxelData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Check if segment is marked for clear
	if(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::MARKED_FOR_CLEAR)
	{
		// Segment is marked for clear, clear it
		gVoxelData[pageId].dGridVoxPos[pageLocalId] = 0xFFFFFFFF;
		gVoxelData[pageId].dGridVoxNorm[pageLocalId] = 0xFFFFFFFF;
		gVoxelData[pageId].dGridVoxOccupancy[pageLocalId] = 0xFFFFFFFF;
	}
}

__global__ void VoxelClearSignal(CVoxelPage* gVoxelData,
								 const uint32_t numPages)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_SEGMENT_PER_PAGE;
	unsigned int pageLocalSegmentId = globalId % GI_SEGMENT_PER_PAGE;

	// Check if segment is marked for clear
	if(globalId >= numPages * GI_SEGMENT_PER_PAGE) return;
	if(ExpandOnlyOccupation(gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId].packed) == SegmentOccupation::MARKED_FOR_CLEAR)
	{
		gVoxelData[pageId].dSegmentObjData[pageLocalSegmentId] = {0};
	}
	 
}