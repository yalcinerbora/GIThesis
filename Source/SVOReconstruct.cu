#include "GIKernels.cuh"
#include "CSparseVoxelOctree.cuh"

__global__ void SVOReconstruct(CSVONode** svo,
							   const CVoxelPage* gVoxelData)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int pageId = globalId / GI_PAGE_SIZE;
	unsigned int pageLocalId = globalId % GI_PAGE_SIZE;

	CVoxelNormPos voxelPos = gVoxelData[pageId].dGridVoxNormPos[pageLocalId];

	// 
	CSVONode svoNode;
	svoNode = static_cast<unsigned char>(voxelPos.x);
	svo[0][globalId] = svoNode;
}


// Low Populates Texture
__global__ void SVOReconstructTop(CSVONode** svo,
								  cudaTextureObject_t svoTexture)
{

}

// Populates from node to node
__global__ void SVOReconstructMid(CSVONode** svo)
{

}

// Populates from page to node
__global__ void SVOReconstructBottom(CSVONode** svo,
									 const CVoxelPage* gVoxelData)
{

}