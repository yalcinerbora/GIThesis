#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"
#include "CHash.cuh"

inline __device__ CSVOColor AtomicColorAvg(CSVOColor* aColor, CSVOColor color)
{
	float4 colorAdd = UnpackSVOColor(color);
	CSVOColor assumed, old = *aColor;
	do
	{
		assumed = old;
		
		// Atomic color average upto 255 colors
		float4 colorAvg = UnpackSVOColor(assumed);
		float ratio = colorAvg.w / (colorAvg.w + 1.0f);
		if(colorAvg.w < 255.0f)
		{
			colorAvg.x = (ratio * colorAvg.x) + (colorAdd.x / (colorAvg.w + 1.0f));
			colorAvg.y = (ratio * colorAvg.y) + (colorAdd.y / (colorAvg.w + 1.0f));
			colorAvg.z = (ratio * colorAvg.z) + (colorAdd.z / (colorAvg.w + 1.0f));
			colorAvg.w += 1.0f;
		}
		old = atomicCAS(aColor, assumed, PackSVOColor(colorAvg));
	}
	while(assumed != old);
	return old;
}

// Reads from page and constructs bottom level
__global__ void SVOReconstruct(CSVONode** svo,
							   const CVoxelPage* gVoxelData,
							   CVoxelRender** gVoxelRender)
{
	//__shared__ unsigned int sLocationHash[GI_THREAD_PER_BLOCK];
	//__shared__ CSVOColor sColor[GI_THREAD_PER_BLOCK];
	//__shared__ CSVONode sNode[GI_THREAD_PER_BLOCK];

	//unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	//unsigned int pageId = globalId / GI_PAGE_SIZE;
	//unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	//unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	//sLocationHash[threadIdx.x] = 0;
	//sNode[threadIdx.x] = 0;
	//sColor[threadIdx.x] = 0;
	//__syncThreads();

	//// Skip Whole segment if necessary
	//if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	//if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	//// Fetch voxel
	//CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];

	//// Skip voxel if invalid
	//if(voxelNormPos.y == 0xFFFFFFFF) return; // FF Normal means invalid vox

	//// Id Fetch to localize
	//unsigned int voxelRenderId;
	//ushort2 objId;
	//CVoxelObjectType objType;
	//CVoxelIds voxelId = gVoxelData[pageId].dGridVoxIds[pageLocalId];
	//ExpandVoxelIds(voxelRenderId, objId, objType, voxelId);
	//
	//// Construct CSVO Node
	//// Since this is bottom level code it will average local colors and writes it to
	//CVoxelRender render = gVoxelRender[objId.y][voxelRenderId];
	//unsigned int  location = Map(sLocationHash, voxelNormPos.x, GI_THREAD_PER_BLOCK);
	//AtomicColorAvg(sColor + location, render.color);
	//__syncThreads();

	//// Each node has local color average now



	//CSVONode svoNode;
	//svoNode = static_cast<unsigned char>(voxelPos.x);
	//svo[0][globalId] = svoNode;
}

// Low Populates Texture
__global__ void SVOReconstructChildSet(CSVONode* gDenseSVO,
									   const CVoxelPage* gVoxelData,
									   const unsigned int denseDim,
									   const unsigned int denseDepth,
									   const unsigned int totalDepth)
{
	__shared__ unsigned int sLocationHash[GI_THREAD_PER_BLOCK];
	__shared__ CSVONode sNode[GI_THREAD_PER_BLOCK];

	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;
	unsigned int pageLocalSegmentId = pageLocalId / GI_SEGMENT_SIZE;

	// Skip Whole segment if necessary
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::EMPTY) return;
	if(gVoxelData[pageId].dIsSegmentOccupied[pageLocalSegmentId] == SegmentOccupation::MARKED_FOR_CLEAR) assert(false);

	sLocationHash[threadIdx.x] = 0;
	sNode[threadIdx.x] = 0;
	__syncThreads();
	
	// Fetch voxel
	CVoxelNormPos voxelNormPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];
	
	// Skip voxel if invalid
	uint3 voxelPos;
	if(voxelNormPos.y != 0xFFFFFFFF)
	{
		// Hash this levels locations in shared
		// Hash locations are same
		voxelPos = ExpandOnlyVoxPos(voxelNormPos.x);
		unsigned int childBit = 0;
		childBit = ((voxelPos.z >> (totalDepth - denseDepth - 1)) & 0x000000001) << 2;
		childBit |= ((voxelPos.y >> (totalDepth - denseDepth - 1)) & 0x000000001) << 1;
		childBit |= ((voxelPos.x >> (totalDepth - denseDepth - 1)) & 0x000000001) << 0;
		childBit = 0x00000001 << (childBit - 1);

		// Dense voxel parent id
		voxelPos.x = (voxelPos.x >> (totalDepth - denseDepth));
		voxelPos.y = (voxelPos.y >> (totalDepth - denseDepth));
		voxelPos.z = (voxelPos.z >> (totalDepth - denseDepth));

		unsigned int packedVoxLevel = 0;
		packedVoxLevel = voxelPos.z << (denseDepth * 2);
		packedVoxLevel |= voxelPos.y << (denseDepth * 1);
		packedVoxLevel |= voxelPos.x << (denseDepth * 0);

		// Atomic Hash Location find and write
		unsigned int  location = Map(sLocationHash, packedVoxLevel, GI_THREAD_PER_BLOCK);
		atomicOr(sNode + location, PackNode(0, static_cast<unsigned char>(childBit)));
	}

	// Wait everything to be written
	__syncThreads();

	// Kill unwritten table indices
	if(sNode[threadIdx.x] == 0) return;

	// Global write to denseVoxel Array
	atomicOr(gDenseSVO +
			 denseDim * denseDim * voxelPos.z +
			 denseDim * voxelPos.y +
			 voxelPos.z,
			 sNode[threadIdx.x]);
}

__global__ void SVOReconstructAllocateNext(CSVONode* gSparseSVO,
										   const unsigned int denseDim,
										   unsigned int& svoCount,
										   const CSVONode* gDenseSVO)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	if(globalId >= denseDim * denseDim * denseDim) return;

	CSVONode node = gSparseSVO[globalId];
	unsigned int childCount;
	unsigned char childBits;
	UnpackNode(childCount, childBits, node);
	
	childCount = 0;
	#pragma unroll 
	for(unsigned int i = 0; i < 8; i++)
	{
		childCount += childBits >> i & 0x01;
	}

	unsigned int location = atomicAdd(&svoCount, childCount);

	//....
}
										   