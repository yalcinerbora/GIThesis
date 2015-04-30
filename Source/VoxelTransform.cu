#include <cuda.h>
#include <cuda_runtime.h>

#include "COpenGLCommon.cuh"
#include "CVoxel.cuh"

__global__ void VoxelTransform(CVoxelPacked* gVoxelData,
							   CVoxelRender* gVoxelRenderData,
							   const CObjectTransformOGL* gObjTransforms,
							   const CVoxelGrid& globalVoxel)
{
	unsigned int globalId;
	uint3 voxPos;
	float3 normal;
	unsigned int objectId;
	bool outOfBounds;

	// Mem Fetch and Expand (8 byte per warp, coalesced, 0 stride)
	ExpandVoxelData(voxPos, objectId, gVoxelData[globalId]);

	// Non static object. Transform voxel & normal
	// Generate World Position
	float4 worldPos;
	worldPos.x = globalVoxel.position.x + voxPos.x * globalVoxel.dimentions.w;
	worldPos.y = globalVoxel.position.y + voxPos.y * globalVoxel.dimentions.w;
	worldPos.z = globalVoxel.position.z + voxPos.z * globalVoxel.dimentions.w;
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
	float invSpan = 1.0f / globalVoxel.dimentions.w;
	worldPos.x -= globalVoxel.position.x;
	worldPos.y -= globalVoxel.position.y;
	worldPos.z -= globalVoxel.position.z;

	outOfBounds = worldPos.x < 0;
	outOfBounds |= worldPos.y < 0;
	outOfBounds |= worldPos.z < 0;

	voxPos.x = static_cast<unsigned int>((worldPos.x -= globalVoxel.position.x) * invSpan);
	voxPos.y = static_cast<unsigned int>((worldPos.y -= globalVoxel.position.y) * invSpan);
	voxPos.z = static_cast<unsigned int>((worldPos.z -= globalVoxel.position.z) * invSpan);

	// Now Write
	if(!outOfBounds)
	{
		PackVoxelData(gVoxelData[globalId], voxPos, objectId);
		gVoxelRenderData[globalId].normal = normal;
	}
}