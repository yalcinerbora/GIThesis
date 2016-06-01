#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CVoxel.cuh"
#include <cuda.h>

// Lookup table for determining neigbour nodes
// just splitted first 8 values
__device__ static const char3 voxLookup[] =
{
	{0, 0, 0},
	{0, 0, 1},
	{0, 1, 0},
	{0, 1, 1},

	{1, 0, 0},
	{1, 0, 1},
	{1, 1, 0},
	{1, 1, 1},
};

inline __device__ CSVOMaterial Average(const CSVOMaterial& material,
									   const float4& colorUnpack,
									   const float3& normalUnpack)
{
	// Unpack Material
	CSVOColor avgColorPacked;
	CVoxelNorm avgNormalPacked;
	UnpackSVOMaterial(avgColorPacked, avgNormalPacked, material);
	float4 avgColor = UnpackSVOColor(avgColorPacked);
	float4 avgNormal = UnpackNormCount(avgNormalPacked);

	// Averaging (color.w is number of nodes)
	assert(avgNormal.w < 255.0f);
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

	avgColorPacked = PackSVOColor(avgColor);
	avgNormalPacked = PackNormCount(avgNormal);
	return PackSVOMaterial(avgColorPacked, avgNormalPacked);
}

inline __device__ CSVOMaterial AddMat(const CSVOMaterial& material,
									  const float4& colorUnpack,
									  const float4& normalUnpack)
{
	// Unpack Material
	CSVOColor avgColorPacked;
	CVoxelNorm avgNormalPacked;
	UnpackSVOMaterial(avgColorPacked, avgNormalPacked, material);
	float4 avgColor = UnpackSVOColor(avgColorPacked);
	float4 avgNormal = ExpandOnlyNormal(avgNormalPacked);

	// Accum Color
	avgColor.x += colorUnpack.x;
	avgColor.y += colorUnpack.y;
	avgColor.z += colorUnpack.z;
	avgColor.w += colorUnpack.w;

	// New Normal Average
	avgNormal.x += normalUnpack.x;
	avgNormal.y += normalUnpack.y;
	avgNormal.z += normalUnpack.z;
	avgNormal.w += normalUnpack.w;

	avgColorPacked = PackSVOColor(avgColor);
	avgNormalPacked = PackOnlyVoxNorm(avgNormal);
	return PackSVOMaterial(avgColorPacked, avgNormalPacked);
}

inline __device__ CSVOMaterial AtomicColorNormalAvg(CSVOMaterial* gMaterial,
													CSVOColor color,
													CVoxelNorm voxelNormal)
{
	float4 colorUnpack = UnpackSVOColor(color);
	float4 normalUnpack = ExpandOnlyNormal(voxelNormal);
	CSVOMaterial assumed, old = *gMaterial;
	do
	{
		assumed = old;
		old = atomicCAS(gMaterial, assumed, Average(assumed,
			colorUnpack,
			{normalUnpack.x, normalUnpack.y, normalUnpack.z}));
	}
	while(assumed != old);
	return old;
}

inline __device__ CSVOMaterial AtomicMatAdd(CSVOMaterial* gMaterial,
											const CSVOColor& color,
											const CVoxelNorm& voxelNormal)
{
	float4 colorUnpack = UnpackSVOColor(color);
	float4 normalUnpack = ExpandOnlyNormal(voxelNormal);
	CSVOMaterial assumed, old = *gMaterial;
	do
	{
		assumed = old;
		old = atomicCAS(gMaterial, assumed, AddMat(assumed,
			colorUnpack,
			normalUnpack));
	}
	while(assumed != old);
	return old;
}

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
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

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
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

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
										   CVoxelRender** gVoxelRenderData,

										   // Constants
										   const unsigned int matSparseOffset,
										   const unsigned int cascadeNo,
										   const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

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
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxNorm[pageLocalId];
	CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][voxelId].color);

	// Atomic Average
	AtomicColorNormalAvg(gSVOMat + matSparseOffset +
						 gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] +
						 nodeIndex,
						 voxelColorPacked,
						 voxelNormPacked);

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

	// Cull if out of range
	if(globalId > levelNodeCount) return;

	// Read Sibling Materials
	const CSVONode* n = (currentLevel == svoConstants.denseDepth) ? gSVODense : gSVOSparse;
	CSVONode node = n[gSVOLevelOffset + globalId];
	
	// Only fetch parent when there is one
	bool fetchParentMat = ((svoConstants.totalDepth - currentLevel) < svoConstants.numCascades);
	CSVOMaterial parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + globalId] : 0;
	
	// Cull if there is no node but continue if there is parent
	if(!fetchParentMat && node == 0xFFFFFFFF) return;

	// Average Portion
	// Material Data
	unsigned int count = 0;
	float4 colorAvg = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 normalAvg = {0.0f, 0.0f, 0.0f, 0.0f};

	// Parent Incorporate
	if(parentMat != 0)
	{
		CSVOColor colorPacked;
		CVoxelNorm normalPacked;
		UnpackSVOMaterial(colorPacked, normalPacked, parentMat);
		float4 color = UnpackSVOColor(colorPacked);
		float4 normal = ExpandOnlyNormal(normalPacked);

		colorAvg.x = 8 * color.x;
		colorAvg.y = 8 * color.y;
		colorAvg.z = 8 * color.z;
		colorAvg.w = 8 * color.w;

		normalAvg.x = 8 * normal.x;
		normalAvg.y = 8 * normal.y;
		normalAvg.z = 8 * normal.z;
		normalAvg.w = 8 * ceil(normal.w);

		count += 8;
	}

	// Average
	if(node != 0xFFFFFFFF)
	{
		#pragma unroll
		for(unsigned int i = 0; i < 8; i++)
		{
			unsigned int nodeId = node + i;
			CSVOMaterial mat = gSVOMat[matOffset + gSVONextLevelOffset + nodeId];
			if(mat == 0) continue;

			CSVOColor colorPacked;
			CVoxelNorm normalPacked;
			UnpackSVOMaterial(colorPacked, normalPacked, mat);
			float4 color = UnpackSVOColor(colorPacked);
			float4 normal = ExpandOnlyNormal(normalPacked);

			colorAvg.x += color.x;
			colorAvg.y += color.y;
			colorAvg.z += color.z;
			colorAvg.w += color.w;

			normalAvg.x += normal.x;
			normalAvg.y += normal.y;
			normalAvg.z += normal.z;
			normalAvg.w += (currentLevel == (svoConstants.totalDepth - 1)) ? ceil(normal.w) : normal.w;

			count++;
		}
	}

	// Divide by Count
	if(count == 0) count = 1.0f;
	float countInv = 1.0f / static_cast<float>(count);
	colorAvg.x *= countInv;
	colorAvg.y *= countInv;
	colorAvg.z *= countInv;
	colorAvg.w *= countInv;

	normalAvg.x *= countInv;
	normalAvg.y *= countInv;
	normalAvg.z *= countInv;
	normalAvg.w *= (count > 8) ? 0.0625f : 0.125f;

	// Pack and Store	
	CSVOColor colorPacked = PackSVOColor(colorAvg);
	CVoxelNorm normPacked = PackOnlyVoxNorm(normalAvg);
	CSVOMaterial matAvg = PackSVOMaterial(colorPacked, normPacked);
	if(currentLevel == svoConstants.denseDepth)
	{
		int3 dim =
		{
			static_cast<int>(globalId % svoConstants.denseDim),
			static_cast<int>((globalId / svoConstants.denseDim) % svoConstants.denseDim),
			static_cast<int>(globalId / (svoConstants.denseDim * svoConstants.denseDim))
		};
		uint2 data =
		{
			static_cast<unsigned int>(matAvg & 0x00000000FFFFFFFF),
			static_cast<unsigned int>(matAvg >> 32)
		};
		surf3Dwrite(data, sDenseMat, dim.x * sizeof(uint2), dim.y, dim.z);
	}
	else
	{
		gSVOMat[matOffset + gSVOLevelOffset + globalId] =  matAvg;
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
	uint2 data;
	surf3Dread(&data, sDenseMatChild,
			   childId3D.x * sizeof(uint2),
			   childId3D.y,
			   childId3D.z);

	// Data
	unsigned int count = (data.x == 0 && data.y == 0) ? 0 : 1;
	float4 color = UnpackSVOColor(data.x);
	float4 normal = ExpandOnlyNormal(data.y);

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
		normal.w += __shfl_down(normal.w, offset, GI_DENSE_WORKER_PER_PARENT);

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
	normal.w *= 0.125f;

	data.x = PackSVOColor(color);
	data.y = PackOnlyVoxNorm(normal);
	if(globalId % GI_DENSE_WORKER_PER_PARENT == 0 && count != 0)
	{
		surf3Dwrite(data, sDenseMatParent,
					parentId3D.x * sizeof(uint2),
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
							   CVoxelRender** gVoxelRenderData,

							   const unsigned int matSparseOffset,
							   const unsigned int cascadeNo,
							   const CSVOConstants& svoConstants)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	// Fetch voxel
	CVoxelPos voxelPosPacked = gVoxelData[pageId].dGridVoxPos[pageLocalId];
	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	uint3 voxelPos = ExpandToSVODepth(voxelUnpacked, cascadeNo,
									  svoConstants.numCascades,
									  svoConstants.totalDepth);

	unsigned int location;
	unsigned int cascadeMaxLevel = svoConstants.totalDepth - (svoConstants.numCascades - cascadeNo);
	for(unsigned int i = svoConstants.denseDepth; i <= cascadeMaxLevel; i++)
	{
		unsigned int levelIndex = i - svoConstants.denseDepth;
		CSVONode* node = nullptr;
		if(i == svoConstants.denseDepth)
		{
			uint3 levelVoxId = CalculateLevelVoxId(voxelPos, i, svoConstants.totalDepth);
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
		unsigned int childId = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
		location += childId;
	}

	ushort2 objectId;
	CVoxelObjectType objType;
	unsigned int voxelId;
	ExpandVoxelIds(voxelId, objectId, objType, gVoxelData[pageId].dGridVoxIds[pageLocalId]);

	CVoxelNorm voxelNormPacked = gVoxelData[pageId].dGridVoxNorm[pageLocalId];
	CSVOColor voxelColorPacked = *reinterpret_cast<unsigned int*>(&gVoxelRenderData[objectId.y][voxelId].color);
	AtomicColorNormalAvg(gSVOMat + matSparseOffset +
						 gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] + location,
						 voxelColorPacked,
						 voxelNormPacked);

	//// Non atmoic overwrite
	//gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 -
	//		svoConstants.denseDepth] +
	//		nodeIndex] = PackSVOMaterial(voxelColorPacked, voxelNormPacked);
}
