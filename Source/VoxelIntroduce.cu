#include <cuda_runtime.h>
#include <cuda.h>

#include "GIKernels.cuh"
#include "CVoxel.cuh"
#include "CAxisAlignedBB.cuh"
#include "COpenGLCommon.cuh"

__global__ void VoxelIntroduce(CVoxelData* gVoxelData,
							   const unsigned int gPageAmount,
							   const CVoxelPacked* gObjectVoxelCache,
							   const CVoxelRender* gObjectVoxelRenderCache,
							   const CObjectTransform& gObjTransform,
							   const CObjectAABB& objAABB,
							   const float objectGridSpan,
							   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;

	// Mem Fetch
	unsigned int objectId;
	uint3 voxPos;
	ExpandVoxelData(voxPos, objectId, gObjectVoxelCache[globalId]);

	// Generate Model Space Position from voxel
	float4 localPos;
	localPos.x = objAABB.min.x + voxPos.x * objectGridSpan;
	localPos.y = objAABB.min.y + voxPos.y * objectGridSpan;
	localPos.z = objAABB.min.z + voxPos.z * objectGridSpan;
	localPos.w = 1.0f;

	// Convert it to world space
	MultMatrixSelf(localPos, gObjTransform.transform);

	// We need to check scaling and adjust span
	// Objects may have different voxel sizes and voxel sizes may change after scaling
	float3 scaling = ExtractScaleInfo(gObjTransform.transform);
	uint3 voxelDim;
	voxelDim.x = static_cast<unsigned int>(objectGridSpan * scaling.x / gGridInfo.span);
	voxelDim.y = static_cast<unsigned int>(objectGridSpan * scaling.y / gGridInfo.span);
	voxelDim.z = static_cast<unsigned int>(objectGridSpan * scaling.z / gGridInfo.span);

	// Discard if voxel is too small
	if(voxelDim.x * voxelDim.y * voxelDim.z == 0.0f) return;

	// We need to construct additional voxels if this voxel spans multiple gird locations
	for(unsigned int i = 0; i < voxelDim.x * voxelDim.y * voxelDim.z; i++)
	{
		float3 localPosSubVox;  
		localPosSubVox.x = localPos.x + (2 * (i % voxelDim.x) - voxelDim.x) * objectGridSpan / 0.5f;
		localPosSubVox.y = localPos.y + (2 * (i % voxelDim.y) - voxelDim.y) * objectGridSpan / 0.5f;
		localPosSubVox.z = localPos.z + (2 * (i % voxelDim.z) - voxelDim.z) * objectGridSpan / 0.5f;

		// For each newly introduced voxel
		// Compare world pos with grid
		// Reconstruct Voxel Indices relative to the new pos of the grid
		localPosSubVox.x -= gGridInfo.position.x;
		localPosSubVox.y -= gGridInfo.position.y;
		localPosSubVox.z -= gGridInfo.position.z;

		bool outOfBounds;
		outOfBounds = (localPosSubVox.x) < 0 || (localPosSubVox.x > gGridInfo.dimension.x * gGridInfo.span);
		outOfBounds |= (localPosSubVox.y) < 0 || (localPosSubVox.x > gGridInfo.dimension.y * gGridInfo.span);
		outOfBounds |= (localPosSubVox.z) < 0 || (localPosSubVox.x > gGridInfo.dimension.z * gGridInfo.span);

		if(!outOfBounds)
		{
			float3 normal = gObjectVoxelRenderCache[globalId].normal;
			MultMatrixSelf(normal, gObjTransform.rotation);

			float invSpan = 1.0f / gGridInfo.span;
			voxPos.x = static_cast<unsigned int>((localPos.x) * invSpan);
			voxPos.y = static_cast<unsigned int>((localPos.y) * invSpan);
			voxPos.z = static_cast<unsigned int>((localPos.z) * invSpan);

			// Determine A Position
			// TODO: Optimize this (template loop unrolling)
			// page size should be adjusted to compensate that (multiples of two)
			// shoudl be like 256 pages at most
			for(unsigned int i = 0; i < gPageAmount; i++)
			{
				// Check this pages empty spaces
				unsigned int location;
				location = atomicDec(&gVoxelData[i].dEmptyElementIndex, 0xFFFFFFFF);
				if(location != 0xFFFFFFFF)
				{
					// Found a Space
					// Write
					PackVoxelData(gVoxelData[i].dGridVoxels[location], voxPos, globalId + 1);
					gVoxelData[i].dVoxelsRenderData[location].normal = normal;
				}
			}
		}
	}
}

__global__ void VoxelObjectCull(unsigned int* gObjectIndices,
								unsigned int& gIndicesIndex,
								const CObjectAABB* gObjectAABB,
								const CObjectTransform* gObjTransforms,
								const CVoxelGrid& globalGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int writeIndex;
	bool intersects = false;

	// Comparing two AABB (Grid Itself is an AABB)
	CAABB gridAABB =
	{
		{globalGridInfo.position.x, globalGridInfo.position.y, globalGridInfo.position.z},
		{
			globalGridInfo.position.x + globalGridInfo.span *globalGridInfo.dimension.x,
			globalGridInfo.position.y + globalGridInfo.span *globalGridInfo.dimension.y,
			globalGridInfo.position.z + globalGridInfo.span *globalGridInfo.dimension.z
		},
	};
	intersects = Intersects(gridAABB, gObjectAABB[globalId]);

	if(intersects)
	{
		writeIndex = atomicAdd(&gIndicesIndex, 0);
		gObjectIndices[writeIndex] = globalId;
	}
}