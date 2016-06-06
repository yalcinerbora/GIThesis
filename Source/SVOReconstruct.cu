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

inline __device__ unsigned int AtomicAllocateNode(CSVONode* gNode,
												  unsigned int& gLevelAllocator,
												  bool isDense,
												  unsigned int* gNodeId,
												  const unsigned int voxLevelId)
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
	
	// Early Check
	if(*gNode < 0xFFFFFFFE) return *gNode;

	CSVONode old = 0xFFFFFFFE;
	while(old == 0xFFFFFFFE)
	{
		old = atomicCAS(gNode, 0xFFFFFFFF, 0xFFFFFFFE);
		if(old == 0xFFFFFFFF)
		{
			// Add Node Id for Average Helping
			if(!isDense) *gNodeId = voxLevelId;

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

extern __global__ void SVOParentHalf(CSVOMaterial* gSVOMat,
									 const CSVONode* gSVOSparse,
									 const unsigned int& gSVOLevelOffset,
									 const unsigned int levelNodeCount,
									 const unsigned int matOffset)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	// Cull if out of range
	if(globalId > levelNodeCount) return;

	// Read
	CSVONode node = gSVOSparse[gSVOLevelOffset + globalId];
	CSVOMaterial mat = gSVOMat[matOffset + gSVOLevelOffset + globalId];

	if(mat == 0x0000000000000000) return;

	// Unpack and Half the Material
	CSVOColor colorPacked;
	CVoxelNorm normalPacked;
	UnpackSVOMaterial(colorPacked, normalPacked, mat);
	float4 normal = ExpandOnlyNormal(normalPacked);
	float4 color = UnpackSVOColor(colorPacked);

	// Check if This Required to be halved
	if(node == 0xFFFFFFFF)
	{
		// Does not have a child no need to be halved
		// However we need to set its opacity to 1	
		// Opacity byte is used to hold atomic average counter
		// If it have atleast 1 voxel it should have full opacity
		normal.w = ceil(normal.w);
	}
	else
	{
		// Have children
		// Half the parent for incoming reduction
		color.x *= 0.5f;
		color.y *= 0.5f;
		color.z *= 0.5f;
		color.w *= 0.5f;

		normal.x *= 0.5f;
		normal.y *= 0.5f;
		normal.z *= 0.5f;
		normal.w = 1.0f;
	}
	mat = PackSVOMaterial(PackSVOColor(color), PackOnlyVoxNorm(normal));
	gSVOMat[matOffset + gSVOLevelOffset + globalId] = mat;
}

__global__ void SVOReconstructAverageNode(CSVOMaterial* gSVOMat,
										  cudaSurfaceObject_t sDenseMat,

										  const CSVONode* gSVODense,
										  const CSVONode* gSVOSparse,
										  const unsigned int* gNodeID,

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
	unsigned int voxelPosPacked = gNodeID[gSVOLevelOffset + globalId];

	// Cull if this is not allocated node
	//if(voxelPosPacked == 0xFFFFFFFF) return;
	if(node == 0xFFFFFFFF) return;

	// Only fetch parent when there is one
	bool fetchParentMat = ((svoConstants.totalDepth - currentLevel) < svoConstants.numCascades);
	CSVOMaterial parentMat = fetchParentMat ? gSVOMat[matOffset + gSVOLevelOffset + globalId] : 0;

	// Average Portion
	// Material Data
	unsigned int count = 0;
	float4 colorAvg = {0.0f, 0.0f, 0.0f, 0.0f};
	float4 normalAvg = {0.0f, 0.0f, 0.0f, 0.0f};

	// Average
	if(node != 0xFFFFFFFF)
	{
		// Node has children
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

		// Divide by Count
		float nominator = (currentLevel == svoConstants.denseDepth) ? 1.0f : 0.125f;
		float countInv = nominator / static_cast<float>(count);
		if(parentMat != 0) countInv *= 0.5f;

		colorAvg.x *= countInv;
		colorAvg.y *= countInv;
		colorAvg.z *= countInv;
		colorAvg.w *= countInv;

		normalAvg.x *= countInv;
		normalAvg.y *= countInv;
		normalAvg.z *= countInv;
		normalAvg.w *= 0.125f * 0.125f;
		normalAvg.w = (parentMat == 0) ? normalAvg.w : 0.125f;
	}
	else
	{
		// Node does not have a children
		// Which means its generated node
		// White color zero spec
		// no occlusion no normal
		float contribution = (parentMat == 0) ? 0.125f : 0.0625f;
		colorAvg.x = contribution;
		colorAvg.y = contribution;
		colorAvg.z = contribution;
		colorAvg.w = 0.0f;

		normalAvg.x = 0.0f;
		normalAvg.y = 0.0f;
		normalAvg.z = 0.0f;
		normalAvg.w = 0.0f;
	}

	// Pack and Store	
	CSVOColor colorPacked = PackSVOColor(colorAvg);
	CVoxelNorm normPacked = PackOnlyVoxNorm(normalAvg);
	if(count != 0)
	{
		if(currentLevel == svoConstants.denseDepth)
		{
			// Directly Write here interpolation occurs differently there
			CSVOMaterial matAvg = PackSVOMaterial(colorPacked, normPacked);
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
			AtomicMatAdd(gSVOMat + gSVOLevelOffset + globalId,
						 colorPacked, normPacked);

			// Add other neigbours
			uint3 voxelUnpacked = UnpackNodeId(voxelPosPacked,
												currentLevel,
												svoConstants.numCascades,
												svoConstants.totalDepth);
			int3 splitId;
			splitId.x = static_cast<int>(voxelUnpacked.x & 0x00000001) * 2 - 1;
			splitId.y = static_cast<int>(voxelUnpacked.y & 0x00000001) * 2 - 1;
			splitId.z = static_cast<int>(voxelUnpacked.z & 0x00000001) * 2 - 1;

			for(unsigned int j = 1; j < 8; j++)
			{
				int3 voxSigned;
				voxSigned.x = static_cast<int>(voxelUnpacked.x) + (voxLookup[j].x * splitId.x);
				voxSigned.y = static_cast<int>(voxelUnpacked.y) + (voxLookup[j].y * splitId.y);
				voxSigned.z = static_cast<int>(voxelUnpacked.z) + (voxLookup[j].z * splitId.z);

				// It may be out of bounds
				int totalDim = 0x1 << currentLevel;
				if(voxSigned.x < 0 || voxSigned.x >= totalDim ||
				   voxSigned.y < 0 || voxSigned.y >= totalDim ||
				   voxSigned.z < 0 || voxSigned.z >= totalDim)
				   continue;

				uint3 vox;
				vox.x = static_cast<unsigned int>(voxSigned.x);
				vox.y = static_cast<unsigned int>(voxSigned.y);
				vox.z = static_cast<unsigned int>(voxSigned.z);

				vox.x <<= (svoConstants.totalDepth - currentLevel);
				vox.y <<= (svoConstants.totalDepth - currentLevel);
				vox.z <<= (svoConstants.totalDepth - currentLevel);

				// Traverse to this node
				unsigned int location;
				for(unsigned int i = svoConstants.denseDepth; i < currentLevel; i++)
				{
					unsigned int levelIndex = i - svoConstants.denseDepth;
					const CSVONode* node = nullptr;
					if(i == svoConstants.denseDepth)
					{
						uint3 levelVoxId = CalculateLevelVoxId(vox, i, svoConstants.totalDepth);
						node = gSVODense +
								svoConstants.denseDim * svoConstants.denseDim * levelVoxId.z +
								svoConstants.denseDim * levelVoxId.y +
								levelVoxId.x;
					}
					else
					{
						node = gSVOSparse + gLevelOffsets[levelIndex] + location;
					}
					location = *node;

					//assert(location != 0xFFFFFFFF);
					if(location == 0xFFFFFFFF) break;

					// Offset child
					unsigned int childId = CalculateLevelChildId(vox, i + 1, svoConstants.totalDepth);
					location += childId;
				}

				if(location != 0xFFFFFFFF)
				{
					AtomicMatAdd(gSVOMat + gSVOLevelOffset + location,
								 colorPacked, normPacked);
				}
				else
				{
					AtomicMatAdd(gSVOMat + gSVOLevelOffset + globalId,
								 colorPacked, normPacked);
				}
			}
		}
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
	if(globalId % GI_DENSE_WORKER_PER_PARENT == 0 &&
	   count != 0)
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
							   unsigned int* gNodeIds,
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
	int3 splitId;
	splitId.x = static_cast<int>(voxelUnpacked.x & 0x00000001) * 2 - 1;
	splitId.y = static_cast<int>(voxelUnpacked.y & 0x00000001) * 2 - 1;
	splitId.z = static_cast<int>(voxelUnpacked.z & 0x00000001) * 2 - 1;

	// Put the color value to the each node corners of the interpolate nodes
	for(unsigned int j = 0; j < 8; j++)
	{
		int3 voxSigned;
		voxSigned.x = static_cast<int>(voxelUnpacked.x) + (voxLookup[j].x * splitId.x);
		voxSigned.y = static_cast<int>(voxelUnpacked.y) + (voxLookup[j].y * splitId.y);
		voxSigned.z = static_cast<int>(voxelUnpacked.z) + (voxLookup[j].z * splitId.z);

		// It may be out of bounds
		int totalDim = 0x1 << (svoConstants.totalDepth - (svoConstants.numCascades - 1));
		if(voxSigned.x < 0 || voxSigned.x >= totalDim ||
		   voxSigned.y < 0 || voxSigned.y >= totalDim ||
		   voxSigned.z < 0 || voxSigned.z >= totalDim)
		   continue;

		uint3 vox;
		vox.x = static_cast<unsigned int>(voxSigned.x);
		vox.y = static_cast<unsigned int>(voxSigned.y);
		vox.z = static_cast<unsigned int>(voxSigned.z);
		uint3 voxelPos = ExpandToSVODepth(vox,
										  cascadeNo,
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
			unsigned int levelNodeId = PackNodeId(voxelPos, i, svoConstants.numCascades, svoConstants.totalDepth);
			assert(levelNodeId != 0xFFFFFFFF);
			location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1],
										  i == svoConstants.denseDepth,
										  gNodeIds + gLevelOffsets[levelIndex] + location,
										  levelNodeId);
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

		// Write this Id also
		if(cascadeMaxLevel + 1 != svoConstants.totalDepth)
		{
			unsigned int nodeLoc = gLevelOffsets[cascadeMaxLevel + 1 - svoConstants.denseDepth] + location;
			gNodeIds[nodeLoc] = PackNodeId(voxelPos,
										   cascadeMaxLevel + 1,
										   svoConstants.numCascades,
										   svoConstants.totalDepth);
		}

		//// Non atmoic overwrite
		//gSVOMat[matSparseOffset + gLevelOffsets[cascadeMaxLevel + 1 -
		//		svoConstants.denseDepth] +
		//		nodeIndex] = PackSVOMaterial(voxelColorPacked, voxelNormPacked);
	}
}

__global__ void SVOReconstruct(CSVONode* gSVOSparse,
							   CSVONode* gSVODense,
							   unsigned int* gNodeIds,
							   unsigned int* gLevelAllocators,

							   const unsigned int& gSVOLevelOffset,
							   const unsigned int* gLevelOffsets,
							   const unsigned int* gLevelTotalSizes,

							   const unsigned int levelNodeCount,
							   const unsigned int levelNo,
							   const CSVOConstants& svoConstants)
{
	// Logic is Per node in the level
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId > levelNodeCount) return;

	// Read Node and its id
	CSVONode node = gSVOSparse[gSVOLevelOffset + globalId];
	unsigned int voxelPosPacked = gNodeIds[gSVOLevelOffset + globalId];

	if(voxelPosPacked == 0xFFFFFFFF) return;

	// Local Voxel pos and expand it if its one of the inner cascades
	uint3 voxelUnpacked = ExpandOnlyVoxPos(voxelPosPacked);
	int3 splitId;
	splitId.x = static_cast<int>(voxelUnpacked.x & 0x00000001) * 2 - 1;
	splitId.y = static_cast<int>(voxelUnpacked.y & 0x00000001) * 2 - 1;
	splitId.z = static_cast<int>(voxelUnpacked.z & 0x00000001) * 2 - 1;

	// Traverse to the each neigbour
	for(unsigned int j = 0; j < 8; j++)
	{
		int3 voxSigned;
		voxSigned.x = static_cast<int>(voxelUnpacked.x) + (voxLookup[j].x * splitId.x);
		voxSigned.y = static_cast<int>(voxelUnpacked.y) + (voxLookup[j].y * splitId.y);
		voxSigned.z = static_cast<int>(voxelUnpacked.z) + (voxLookup[j].z * splitId.z);

		// It may be out of bounds
		int totalDim = 0x1 << (svoConstants.totalDepth - (svoConstants.numCascades - 1));
		if(voxSigned.x < 0 || voxSigned.x >= totalDim ||
		   voxSigned.y < 0 || voxSigned.y >= totalDim ||
		   voxSigned.z < 0 || voxSigned.z >= totalDim)
		   continue;

		uint3 vox;
		vox.x = static_cast<unsigned int>(voxSigned.x);
		vox.y = static_cast<unsigned int>(voxSigned.y);
		vox.z = static_cast<unsigned int>(voxSigned.z);
		uint3 voxelPos = ExpandToSVODepth(vox,
										  levelNo,
										  svoConstants.numCascades,
										  svoConstants.totalDepth);

		unsigned int location;
		for(unsigned int i = svoConstants.denseDepth; i <= levelNo; i++)
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
			unsigned int levelNodeId = PackNodeId(voxelPos, i, svoConstants.numCascades, svoConstants.totalDepth);
			assert(levelNodeId != 0xFFFFFFFF);
			location = AtomicAllocateNode(node, gLevelAllocators[levelIndex + 1],
										  i == svoConstants.denseDepth,
										  gNodeIds + gLevelOffsets[levelIndex] + location,
										  levelNodeId);
			assert(location < gLevelTotalSizes[levelIndex + 1]);

			// Offset child
			unsigned int childId = CalculateLevelChildId(voxelPos, i + 1, svoConstants.totalDepth);
			location += childId;
		}
	}
}