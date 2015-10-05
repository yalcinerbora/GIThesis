/**

Sparse voxel octree implementation
Designed for fast reconstruction from its bottom 

*/

#ifndef __CSPARSEVOXELOCTREE_H__
#define __CSPARSEVOXELOCTREE_H__

#include <cuda_runtime.h>
#include <cuda.h>
#include "CVoxel.cuh"


#pragma pack(push, 1)
struct CSVONode
{
	unsigned short	index;
	unsigned char	childMarkup;
};
#pragma pack(pop)

// Returns the intersected voxelIndex if voxel is found
// Else it returns -1 (0xFFFFFFFF)
__global__ unsigned int FindIntersection(const CVoxelGrid& gGridInfo,
										 const CSVONode* root,
										 const float3& worldPos)
{
	unsigned int depth = 1;
	unsigned int voxelIndex = 0;

	const CSVONode* currentNode = root;
	while(currentNode != nullptr || depth <= gGridInfo.depth)
	{
		float3 childPos;
		float invDepth = gGridInfo.span * 2 * depth;
		childPos.x = worldPos.x - gGridInfo.dimension.x * invDepth;
		childPos.y = worldPos.y - gGridInfo.dimension.y * invDepth;
		childPos.z = worldPos.z - gGridInfo.dimension.z * invDepth;

		// Determine the next child
		unsigned int nextChildId = 0;
		nextChildId |= 0x00000001 && worldPos.x > childPos.x;
		nextChildId |= 0x00000002 && worldPos.y > childPos.y;
		nextChildId |= 0x00000004 && worldPos.z > childPos.z;

		// Find Child
		// We cant directly hop to the next data since tree is sparse
		// Iterate child array
		for(unsigned int i = 0; i < currentNode->childCount; i++)
		{
			if(currentNode->childPtr[i].childId == nextChildId)
			{
				currentNode = &currentNode->childPtr[i];
				depth++;
			}
		}
	}
	return (currentNode == nullptr) ? reinterpret_cast<unsigned int>(currentNode) : -1;
}
#endif //__CSPARSEVOXELOCTREE_H__