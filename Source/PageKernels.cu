#include <cuda_runtime.h>
#include <cuda.h>
#include <assert.h>

#include "PageKernels.cuh"
#include "CVoxelFunctions.cuh"
#include "CMatrixFunctions.cuh"
#include "CAABBFunctions.cuh"
#include "COpenGLTypes.h"
#include "CAtomicPageAlloc.cuh"
#include "GIVoxelPages.h"

#define GI_MAX_JOINT_COUNT 63

__global__ void InitializePage(unsigned char* emptySegments, const size_t pageCount)
{
	size_t sizePerPage = GIVoxelPages::PageSize *
						 (sizeof(CVoxelPos) +
						  sizeof(CVoxelNorm) +
						  sizeof(CVoxelOccupancy))
						 +
						 GIVoxelPages::SegmentSize *
						 (sizeof(unsigned char) +
						  sizeof(CSegmentInfo));

	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageLocalSegmentId = globalId % GIVoxelPages::SegmentPerPage;
	unsigned int pageId = globalId / GIVoxelPages::SegmentPerPage;

	// Cull if out of bounds
	if(globalId >= pageCount * GIVoxelPages::SegmentPerPage) return;
	emptySegments[pageId * sizePerPage + pageLocalSegmentId] = GIVoxelPages::SegmentPerPage - pageLocalSegmentId - 1;
}

inline __device__ unsigned int WarpAggragateIndex(unsigned int& gAtomicIndex)
{
	unsigned int activeThreads = __ballot(0x1);
	unsigned int incrementCount = __popc(activeThreads);
	unsigned int leader = __ffs(activeThreads) - 1;
	unsigned int warpLocalId = threadIdx.x % warpSize;

	unsigned int baseIndex;
	if(warpLocalId == leader)
		baseIndex = atomicAdd(&gAtomicIndex, incrementCount);
	baseIndex = __shfl(baseIndex, leader);
	return baseIndex + __popc(activeThreads & ((1 << warpLocalId) - 1));
}

__global__ void CopyPage(// OGL Buffer
						 VoxelPosition* gVoxelPosition,
						 unsigned int* gVoxelRender,
						 unsigned int& gAtomicIndex,
						 // Voxel Cache
						 const BatchVoxelCache* gBatchVoxelCache,
						 // Voxel Pages
						 const CVoxelPageConst* gVoxelPages,
						 //
						 const uint32_t batchCount,
						 const uint32_t selectedCascade,
						 const VoxelRenderType renderType)
{
	// Shared Memory for generic data
	__shared__ CSegmentInfo sSegInfo;
	__shared__ CMeshVoxelInfo sMeshVoxelInfo;

	unsigned int blockLocalId = threadIdx.x;
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
		// Load to shred memory
		sSegInfo = gVoxelPages[pageId].dSegmentInfo[pageLocalSegmentId];
		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	}
	__syncthreads();	
	if(blockLocalId != 0)
	{
		ExpandSegmentInfo(cascadeId, objType, occupation, firstOccurance, sSegInfo.packed);
	}

	// Full Block Cull
	if(cascadeId < selectedCascade) return;
	if(occupation == CSegmentOccupation::EMPTY) return;
	assert(occupation != CSegmentOccupation::MARKED_FOR_CLEAR);

	if(blockLocalId == 0)
	{
		sMeshVoxelInfo = gBatchVoxelCache[cascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId];
	}
	__syncthreads();

	// Now Copy If individual voxel is valid
	CVoxelNorm voxNorm = gVoxelPages[pageId].dGridVoxNorm[pageLocalId];
	if(voxNorm != 0xFFFFFFFF)
	{
		// Get Index
		unsigned int index = atomicAdd(&gAtomicIndex, 1);
		//unsigned int index = WarpAggragateIndex(gAtomicIndex);

		// Get Data
		if(renderType != VoxelRenderType::NORMAL)
		{
			// Find your opengl data and voxel cache
			// then find appropriate albedo
			const uint16_t& batchId = sSegInfo.batchId;
			const BatchVoxelCache& batchCache = gBatchVoxelCache[cascadeId * batchCount + batchId];
			const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
			const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelInfo.voxOffset;

			voxNorm = batchCache.dVoxelAlbedo[batchLocalVoxelId];
		}

		// Inject Voxel Pos
		CVoxelPos voxPos = gVoxelPages[pageId].dGridVoxPos[pageLocalId];
		voxPos |= (cascadeId & 0x00000003) << 30;

		gVoxelPosition[index] = voxPos;
		gVoxelRender[index] = voxNorm;
	}
}

__global__ void VoxelDeallocate(// Voxel System
								CVoxelPage* gVoxelPages,
								const CVoxelGrid* gGridInfos,
								// Helper Structures								  
								ushort2* gSegmentAllocInfo,
								const CSegmentInfo* gSegmentInfo,
								// Per Object Related
								const BatchOGLData* gBatchOGLData,
								// Limits
								const uint32_t totalSegments)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Now Thread Scheme changes per objectSegment
	if(globalId >= totalSegments) return;

	// Unpack segmentInfo
	const CSegmentInfo segInfo = gSegmentInfo[globalId];
	uint8_t cascadeNo = ExpandOnlyCascadeNo(segInfo.packed);
	const CVoxelGrid cascadeGrid = gGridInfos[cascadeNo];

	// Intersection Check
	const uint32_t transformId = gBatchOGLData[segInfo.batchId].dModelTransformIndices[segInfo.objId];
	const CMatrix4x4 transform = gBatchOGLData[segInfo.batchId].dModelTransforms[transformId].transform;
	const CAABB objAABB = gBatchOGLData[segInfo.batchId].dAABBs[segInfo.objId];
	bool intersects = CheckGridVoxIntersect(cascadeGrid, objAABB, transform);

	// Check if this object is not allocated
	ushort2 objAlloc = gSegmentAllocInfo[globalId];
	if(!intersects && objAlloc.x != 0xFFFF)
	{
		// "Dealocate"
		assert(ExpandOnlyOccupation(gVoxelPages[objAlloc.x].dSegmentInfo[objAlloc.y].packed) == CSegmentOccupation::OCCUPIED);
		unsigned int size = AtomicDealloc(&(gVoxelPages[objAlloc.x].dEmptySegmentStackSize), GIVoxelPages::SegmentPerPage);
		assert(size != GIVoxelPages::SegmentPerPage);
		if(size != GIVoxelPages::SegmentPerPage)
		{
			unsigned int location = size;
			gVoxelPages[objAlloc.x].dEmptySegmentPos[location] = objAlloc.y;

			CSegmentInfo segObjId = {};
			gVoxelPages[objAlloc.x].dSegmentInfo[objAlloc.y] = segObjId;
			gSegmentAllocInfo[globalId] = ushort2{0xFFFF, 0xFFFF};
		}
	}
}

__global__ void VoxelAllocate(// Voxel System
							  CVoxelPage* gVoxelPages,
							  const CVoxelGrid* gGridInfos,
							  // Helper Structures
							  ushort2* gSegmentAllocInfo,
							  const CSegmentInfo* gSegmentInfo,
							  // Per Object Related
							  const BatchOGLData* gBatchOGLData,
							  // Limits
							  const uint32_t totalSegments,
							  const uint32_t pageAmount)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Now Thread Scheme changes per objectSegment
	if(globalId >= totalSegments) return;

	// Unpack segmentInfo
	const CSegmentInfo segInfo = gSegmentInfo[globalId];
	uint8_t cascadeNo = ExpandOnlyCascadeNo(segInfo.packed);
	const CVoxelGrid cascadeGrid = gGridInfos[cascadeNo];

	// Intersection Check
	const uint32_t transformId = gBatchOGLData[segInfo.batchId].dModelTransformIndices[segInfo.objId];
	const CMatrix4x4 transform = gBatchOGLData[segInfo.batchId].dModelTransforms[transformId].transform;
	const CAABB objAABB = gBatchOGLData[segInfo.batchId].dAABBs[segInfo.objId];
	bool intersects = CheckGridVoxIntersect(cascadeGrid, objAABB, transform);

	// Check if this object is not allocated
	ushort2 objAlloc = gSegmentAllocInfo[globalId];

	if(intersects && objAlloc.x == 0xFFFF)
	{
		// "Allocate"
		// Check page by page
		for(unsigned int i = 0; i < pageAmount; i++)
		{
			unsigned int size = AtomicAlloc(&(gVoxelPages[i].dEmptySegmentStackSize));
			if(size != 0)
			{
				unsigned int location = gVoxelPages[i].dEmptySegmentPos[size - 1];
				assert(ExpandOnlyOccupation(gVoxelPages[i].dSegmentInfo[location].packed) == CSegmentOccupation::EMPTY);
				gSegmentAllocInfo[globalId] = ushort2
				{
					static_cast<unsigned short>(i),
					static_cast<unsigned short>(location)
				};
				gVoxelPages[i].dSegmentInfo[location] = segInfo;
				return;
			}
		}
	}
}

inline __device__ void LoadTransformData(// Shared Mem
										 CMatrix4x4* sTransformMatrices,
										 CMatrix3x3* sRotationMatrices,
										 uint8_t* sMatrixLookup,
										 // Object Transform Matrix
										 const BatchOGLData& gBatchOGLData,
										 // Current Voxel Weight
										 const uchar4& voxelWeightIndex,
										 // Object Type that will be broadcasted
										 const CObjectType& objType,
										 const uint16_t& objId,
										 const uint16_t& transformId)
{
	unsigned int blockLocalId = threadIdx.x;

	// Here we will load transform and rotation matrices
	// Each thread will load 1 float. There is two 4x4 matrix
	// 32 floats will be loaded
	// Just enough for a warp to do the work
	// Load matrices (4 byte load by each thread sequential no bank conflict)
	const CModelTransform& objectMT = gBatchOGLData.dModelTransforms[transformId];
	float* sTrans = reinterpret_cast<float*>(&sTransformMatrices[0]);
	float* sRot = reinterpret_cast<float*>(&sRotationMatrices[0]);
	if(blockLocalId < 16)
	{
		const float* objectTransform = reinterpret_cast<const float*>(&objectMT.transform);
		sTrans[blockLocalId] = objectTransform[blockLocalId];
	}
	else if(blockLocalId < 25)
	{
		unsigned int rotationId = blockLocalId - 16;
		unsigned int columnId = rotationId / 3;
		unsigned int rowId = rotationId % 3;

		const float* objectRotation = reinterpret_cast<const float*>(&objectMT.rotation);
		sRot[columnId * 4 + rowId] = objectRotation[columnId * 4 + rowId];
	}

	// Load Joint Transforms if Skeletal Object
	if(objType == CObjectType::SKEL_DYNAMIC)
	{
		// All valid objects will request matrix load
		// then entire block will try to load it
		// Max skeleton bone count is 64
		// Worst case 64 * 16 = 1024 float will be loaded to sMem
		// Some blocks will load twice
		// However its extremely rare (even impossible case)
		// In a realistic scenario (and if a segment holds adjacent voxels)
		// And if max bone influence per vertex is around 4 
		// there should be at most 8

		// Matrix Lookup Initialize
		if(blockLocalId < GI_MAX_JOINT_COUNT)
			sMatrixLookup[blockLocalId] = 0;
		__syncthreads();

		if(voxelWeightIndex.x != 0xFF) sMatrixLookup[voxelWeightIndex.x] = 1;
		if(voxelWeightIndex.y != 0xFF) sMatrixLookup[voxelWeightIndex.y] = 1;
		if(voxelWeightIndex.z != 0xFF) sMatrixLookup[voxelWeightIndex.z] = 1;
		if(voxelWeightIndex.w != 0xFF) sMatrixLookup[voxelWeightIndex.w] = 1;
		__syncthreads();

		// Lookup Tables are Loaded
		// Theorethical 63 Matrices will be loaded
		//	Each thread will load 1 float we need 1024 threads
		unsigned int iterationCount = (GI_MAX_JOINT_COUNT * 16) / blockDim.x;
		for(unsigned int i = 0; i < iterationCount; i++)
		{
			unsigned int floatId = blockLocalId + (blockDim.x * i);

			// Transformation
			if(floatId <  GI_MAX_JOINT_COUNT * 16)
			{
				unsigned int matrixId = (floatId / 16);
				unsigned int matrixLocalFloatId = floatId % 16;				
				if(sMatrixLookup[matrixId] == 1)
				{
					const CMatrix4x4& jointT = gBatchOGLData.dJointTransforms[matrixId].transform;
					const float* jointTFloat = reinterpret_cast<const float*>(&jointT);
					float* sTrans = reinterpret_cast<float*>(&sTransformMatrices[matrixId + 1]);

					sTrans[matrixLocalFloatId] = jointTFloat[matrixLocalFloatId];
				}
			}
			// Rotation
			if(floatId < GI_MAX_JOINT_COUNT * 9)
			{
				unsigned int matrixId = (floatId / 9);
				unsigned int matrixLocalFloatId = floatId % 9;
				if(sMatrixLookup[matrixId] == 1)
				{
					const CMatrix4x4& jointRot = gBatchOGLData.dJointTransforms[matrixId].rotation;
					const float* jointRotFloat = reinterpret_cast<const float*>(&jointRot);
					float* sRot = reinterpret_cast<float*>(&sRotationMatrices[matrixId + 1]);

					unsigned int column = matrixLocalFloatId / 3;
					unsigned int row = matrixLocalFloatId % 3;
					sRot[column * 4 + row] = jointRotFloat[column * 4 + row];
				}
			}
		}
	}
	// We write to shared mem sync between warps
	__syncthreads();
}

__global__ void VoxelTransform(// Voxel Pages
							   CVoxelPage* gVoxelPages,
							   const CVoxelGrid* gGridInfos,
							   // OGL Related
							   const BatchOGLData* gBatchOGLData,
							   // Voxel Cache Related
							   const BatchVoxelCache* gBatchVoxelCache,
							   // Limits
							   const uint32_t batchCount)
{
	// Cache Loading
	// Shared Memory which used for transform rendering
	__shared__ CMatrix4x4 sTransformMatrices[GI_MAX_JOINT_COUNT + 1];	// First index holds model matrix
	__shared__ CMatrix3x3 sRotationMatrices[GI_MAX_JOINT_COUNT + 1];
	__shared__ uint8_t sMatrixLookup[GI_MAX_JOINT_COUNT + 1];	// Extra 4 Byte for alignment
															// Shared Memory for generic data
	__shared__ CSegmentInfo sSegInfo;
	__shared__ CVoxelGrid sGridInfo;
	__shared__ uint32_t	sObjTransformId;
	__shared__ CMeshVoxelInfo sMeshVoxelInfo;

	unsigned int blockLocalId = threadIdx.x;
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

	// If segment is not empty
	// Load Block Constants
	if(blockLocalId == 0)
	{
		// TODO: Re-write this to be more multi-thread loadable
		sObjTransformId = gBatchOGLData[sSegInfo.batchId].dModelTransformIndices[sSegInfo.objId];
		sMeshVoxelInfo = gBatchVoxelCache[cascadeId * batchCount + sSegInfo.batchId].dMeshVoxelInfo[sSegInfo.objId];
		sGridInfo = gGridInfos[cascadeId];
	}
	__syncthreads();

	// Find your opengl data and voxel cache
	const uint16_t& batchId = sSegInfo.batchId;
	const uint16_t& objectId = sSegInfo.objId;
	const BatchOGLData& batchOGLData = gBatchOGLData[batchId];
	const BatchVoxelCache& batchCache = gBatchVoxelCache[cascadeId * batchCount + batchId];

	// Voxel Ids
	const uint32_t objectLocalVoxelId = sSegInfo.objectSegmentId * GIVoxelPages::SegmentSize + segmentLocalVoxId;
	const uint32_t batchLocalVoxelId = objectLocalVoxelId + sMeshVoxelInfo.voxOffset;

	// Load weights if necessary
	CVoxelWeights weights = {{0x00, 0x00, 0x00, 0x00},{0xFF, 0xFF, 0xFF, 0xFF}};
	if(objectLocalVoxelId < sMeshVoxelInfo.voxCount && objType == CObjectType::SKEL_DYNAMIC)
	{
		weights = batchCache.dVoxelWeight[batchLocalVoxelId];
	}

	// Segment is occupied so load matrices before culling unused warps
	LoadTransformData(// Shared Mem
					sTransformMatrices,
					sRotationMatrices,
					sMatrixLookup,
					// OGL
					batchOGLData,
					// Weight Index
					weights.weightIndex,
					// Object Type that will be broadcasted
					objType,
					objectId,
					sObjTransformId);

	// Cull threads
	// Edge case where last segment do not always full
	if(objectLocalVoxelId >= sMeshVoxelInfo.voxCount)
	{
		gVoxelPages[pageId].dGridVoxPos[pageLocalId] = 0xFFFFFFFF;
		gVoxelPages[pageId].dGridVoxNorm[pageLocalId] = 0xFFFFFFFF;
		return;
	}

	// Fetch NormalPos from cache
	uint3 voxPos;
	float3 normal;
	voxPos = ExpandVoxPos(batchCache.dVoxelPos[batchLocalVoxelId]);
	normal = ExpandVoxNormal(batchCache.dVoxelNorm[batchLocalVoxelId]);

	// Fetch AABB min, transform and span
	float4 objAABBMin = batchOGLData.dAABBs[objectId].min;

	// Generate World Position
	// start with object space position
	float3 worldPos;
	worldPos.x = objAABBMin.x + voxPos.x * sGridInfo.span;
	worldPos.y = objAABBMin.y + voxPos.y * sGridInfo.span;
	worldPos.z = objAABBMin.z + voxPos.z * sGridInfo.span;

	// Joint Transformations
	if(objType == CObjectType::SKEL_DYNAMIC)
	{
		float4 weightUnorm;
		weightUnorm.x = static_cast<float>(weights.weight.x) / 255.0f;
		weightUnorm.y = static_cast<float>(weights.weight.y) / 255.0f;
		weightUnorm.z = static_cast<float>(weights.weight.z) / 255.0f;
		weightUnorm.w = static_cast<float>(weights.weight.w) / 255.0f;

		//if(threadIdx.x == 0)
		//	printf("x %d, y %d, z %d, w %d\n",
		//	weights.weightIndex.x,
		//	weights.weightIndex.y,
		//	weights.weightIndex.z,
		//	weights.weightIndex.w);

		// Nyra Char Related Assert
		//assert(weights.weightIndex.x <= 24);
		//assert(weights.weightIndex.y <= 24);
		//assert(weights.weightIndex.z <= 24);
		//assert(weights.weightIndex.w <= 24);

		float3 pos = {0.0f, 0.0f, 0.0f};
		float3 p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.x + 1]);
		//float3 p = MultMatrix(worldPos, batchOGLData.dJointTransforms[weights.weightIndex.x].transform);
		pos.x += weightUnorm.x * p.x;
		pos.y += weightUnorm.x * p.y;
		pos.z += weightUnorm.x * p.z;

		p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.y + 1]);
		//p = MultMatrix(worldPos, batchOGLData.dJointTransforms[weights.weightIndex.y].transform);
		pos.x += weightUnorm.y * p.x;
		pos.y += weightUnorm.y * p.y;
		pos.z += weightUnorm.y * p.z;

		p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.z + 1]);
		//p = MultMatrix(worldPos, batchOGLData.dJointTransforms[weights.weightIndex.z].transform);
		pos.x += weightUnorm.z * p.x;
		pos.y += weightUnorm.z * p.y;
		pos.z += weightUnorm.z * p.z;

		p = MultMatrix(worldPos, sTransformMatrices[weights.weightIndex.w + 1]);
		//p = MultMatrix(worldPos, batchOGLData.dJointTransforms[weights.weightIndex.w].transform);
		pos.x += weightUnorm.w * p.x;
		pos.y += weightUnorm.w * p.y;
		pos.z += weightUnorm.w * p.z;

		worldPos = pos;

		float3 norm = {0.0f, 0.0f, 0.0f};
		float3 n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.x + 1]);
		norm.x += weightUnorm.x * n.x;
		norm.y += weightUnorm.x * n.y;
		norm.z += weightUnorm.x * n.z;

		n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.y + 1]);
		norm.x += weightUnorm.y * n.x;
		norm.y += weightUnorm.y * n.y;
		norm.z += weightUnorm.y * n.z;

		n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.z + 1]);
		norm.x += weightUnorm.z * n.x;
		norm.y += weightUnorm.z * n.y;
		norm.z += weightUnorm.z * n.z;

		n = MultMatrix(normal, sRotationMatrices[weights.weightIndex.w + 1]);
		norm.x += weightUnorm.w * n.x;
		norm.y += weightUnorm.w * n.y;
		norm.z += weightUnorm.w * n.z;

		normal = norm;
	}

	// Model Transformations
	MultMatrixSelf(worldPos, sTransformMatrices[0]);
	MultMatrixSelf(normal, sRotationMatrices[0]);
	//// Unoptimized Matrix Load
	//CMatrix4x4 transform = gObjTransforms[segObj.batchId][gObjTransformIds[segObj.batchId][segObj.objId]].transform;
	//CMatrix4x4 rotation = gObjTransforms[segObj.batchId][gObjTransformIds[segObj.batchId][segObj.objId]].transform;
	//MultMatrixSelf(worldPos, transform);
	//MultMatrixSelf(normal, rotation);

	// Reconstruct Voxel Indices relative to the new pos of the grid
	worldPos.x -= sGridInfo.position.x;
	worldPos.y -= sGridInfo.position.y;
	worldPos.z -= sGridInfo.position.z;

	bool outOfBounds;
	outOfBounds = (worldPos.x < 0.0f) || (worldPos.x >= (sGridInfo.dimension.x) * sGridInfo.span);
	outOfBounds |= (worldPos.y < 0.0f) || (worldPos.y >= (sGridInfo.dimension.y) * sGridInfo.span);
	outOfBounds |= (worldPos.z < 0.0f) || (worldPos.z >= (sGridInfo.dimension.z) * sGridInfo.span);

	// If its mip dont update inner cascade
	bool inInnerCascade = false;
	if(!firstOccurance) // Only do inner culling if object is not first occurance in hierarchy (base level voxel data of the object
	{
		inInnerCascade = (worldPos.x > (sGridInfo.dimension.x) * sGridInfo.span * 0.25f) &&
						 (worldPos.x < (sGridInfo.dimension.x) * sGridInfo.span * 0.75f);

		inInnerCascade &= (worldPos.y > (sGridInfo.dimension.y) * sGridInfo.span * 0.25f) &&
						  (worldPos.y < (sGridInfo.dimension.y) * sGridInfo.span * 0.75f);

		inInnerCascade &= (worldPos.z > (sGridInfo.dimension.z) * sGridInfo.span * 0.25f) &&
						  (worldPos.z < (sGridInfo.dimension.z) * sGridInfo.span * 0.75f);
	}
	outOfBounds |= inInnerCascade;

	// Voxel Space
	float invSpan = 1.0f / sGridInfo.span;
	voxPos.x = static_cast<unsigned int>(worldPos.x * invSpan);
	voxPos.y = static_cast<unsigned int>(worldPos.y * invSpan);
	voxPos.z = static_cast<unsigned int>(worldPos.z * invSpan);

	// Calculate VoxelWeights
	float3 volumeWeight;
	volumeWeight.x = worldPos.x * invSpan;
	volumeWeight.y = worldPos.y * invSpan;
	volumeWeight.z = worldPos.z * invSpan;

	volumeWeight.x = volumeWeight.x - static_cast<float>(voxPos.x);
	volumeWeight.y = volumeWeight.y - static_cast<float>(voxPos.y);
	volumeWeight.z = volumeWeight.z - static_cast<float>(voxPos.z);

	//volumeWeight.x = 1.0f;
	//volumeWeight.y = 1.0f;
	//volumeWeight.z = 1.0f;

	//uint3 neigbourBits;
	//neigbourBits.x = (volumeWeight.x > 0) ? 1 : 0;
	//neigbourBits.y = (volumeWeight.y > 0) ? 1 : 0;
	//neigbourBits.z = (volumeWeight.z > 0) ? 1 : 0;

	// Outer Bound Check
	outOfBounds |= (voxPos.x >= sGridInfo.dimension.x);
	outOfBounds |= (voxPos.y >= sGridInfo.dimension.y);
	outOfBounds |= (voxPos.z >= sGridInfo.dimension.z);

	// Now Write
	// Discard the out of bound voxels
	//outOfBounds = false;
	if(!outOfBounds)
	{	
		// Write to page
		gVoxelPages[pageId].dGridVoxPos[pageLocalId] = PackVoxPos(voxPos);
		gVoxelPages[pageId].dGridVoxNorm[pageLocalId] = PackVoxNormal(normal);
		gVoxelPages[pageId].dGridVoxOccupancy[pageLocalId] = PackOccupancy(volumeWeight);
	}
	else
	{
		gVoxelPages[pageId].dGridVoxPos[pageLocalId] = 0xFFFFFFFF;
		gVoxelPages[pageId].dGridVoxNorm[pageLocalId] = 0xFFFFFFFF;
	}
}