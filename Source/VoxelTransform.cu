#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"

__global__ void VoxelTransform(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,
							   unsigned int* gEmptyMarkArray,
							   unsigned int& gEmptyMarkIndex,
							   const CObjectTransformOGL* gObjTransforms,
							   const CVoxelGrid& gGridInfo)
{
	unsigned int globalId = threadIdx.x + blockIdx.x * blockDim.x;
	uint3 voxPos;
	float3 normal;
	unsigned int objectId;
	bool outOfBounds;

	// Mem Fetch and Expand (8 byte per warp, coalesced, 0 stride)
	ExpandVoxelData(voxPos, objectId, gVoxelData[globalId]);

	// Non static object. Transform voxel & normal
	// Generate World Position
	float4 worldPos;
	worldPos.x = gGridInfo.position.x + voxPos.x * gGridInfo.position.w;
	worldPos.y = gGridInfo.position.y + voxPos.y * gGridInfo.position.w;
	worldPos.z = gGridInfo.position.z + voxPos.z * gGridInfo.position.w;
	worldPos.w = 1.0f;

	// Normal Fetch (12 byte per wrap, 4 byte stride)
	normal = gVoxelRenderData[globalId].normal;

	// Transformations (Fetch Irrelevant because of caching)
	// Obj count <<< Voxel Count, Rigid Objects' voxel adjacent each other
	// There is a hight chance that this transform is on the shared mem(cache) already
	CMatrix4x4 transform = gObjTransforms[objectId].transform;
	CMatrix3x3 rotation = gObjTransforms[objectId].rotation;
	MultMatrixSelf(worldPos, transform);
	MultMatrixSelf(normal, rotation);

	// Reconstruct Voxel Indices
	worldPos.x -= gGridInfo.position.x;
	worldPos.y -= gGridInfo.position.y;
	worldPos.z -= gGridInfo.position.z;

	outOfBounds = (worldPos.x) < 0 || (worldPos.x > gGridInfo.dimension.x * gGridInfo.span);
	outOfBounds |= (worldPos.y) < 0 || (worldPos.x > gGridInfo.dimension.y * gGridInfo.span);
	outOfBounds |= (worldPos.z) < 0 || (worldPos.x > gGridInfo.dimension.z * gGridInfo.span);

	float invSpan = 1.0f / gGridInfo.span;
	voxPos.x = static_cast<unsigned int>((worldPos.x) * invSpan);
	voxPos.y = static_cast<unsigned int>((worldPos.y) * invSpan);
	voxPos.z = static_cast<unsigned int>((worldPos.z) * invSpan);

	// Now Write
	// Discard the out of bound voxel since other kernel calls will introduce if the obj
	// will come back into the grid
	if(!outOfBounds)
	{
		PackVoxelData(gVoxelData[globalId], voxPos, objectId);
		gVoxelRenderData[globalId].normal = normal;
	}
	else
	{
		// Mark this node as deleted so that voxel introduce kernel
		// Adds a voxel to this node
		unsigned int index = atomicAdd(&gEmptyMarkIndex, 1);
		gEmptyMarkArray[index] = globalId;
	}
}