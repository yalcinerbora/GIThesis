#version 430
				
// Definitions
#define LU_VOXEL layout(std430, binding = 0)
#define LU_VOXEL_RENDER layout(std430, binding = 1)
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)
#define U_VOX_SLICE layout(location = 5)

#define I_VOX_READ layout(rgba32f, binding = 2) restrict readonly

// I-O
U_OBJ_ID uniform uint objId;
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_VOX_SLICE uniform uvec2 voxSlice;


LU_VOXEL buffer VoxelArray
{
	uvec2 voxelPacked[];
};

LU_VOXEL_RENDER buffer VoxelArrayRender
{
	struct
	{
		vec3 normal;
		uint color;
	} voxelArrayRender[];
};

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

uniform I_VOX_READ image3D voxelData;

uvec2 PackVoxelData(in uvec3 voxCoord, in uint objId)
{
	uvec2 vec;
	vec.x = voxCoord.x;
	vec.x |= voxCoord.y << 16;
	vec.y = voxCoord.z;
	vec.y |= objId << 16;
	return vec;
}
		
layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(void)
{
	uvec3 voxId;
	uint localBlockID = gl_WorkGroupID.x % voxSlice.y;
	voxId.xy  = gl_LocalInvocationID.xy + 
				uvec2( localBlockID % voxSlice.x, localBlockID / voxSlice.x ) * uvec2(32); 
	voxId.z = gl_WorkGroupID.x / voxSlice.y;

	if(voxId.x >= voxDim.x || 
		voxId.y >= voxDim.y ||
		voxId.z >= voxDim.z) return;

	if(gl_GlobalInvocationID.x == 0 &&
		gl_GlobalInvocationID.y == 0)
		atomicExchange(objectGridInfo[objId].voxCount, 0);

	// Force Sync
	memoryBarrier();

	vec4 voxData = imageLoad(voxelData, ivec3(voxId));

	// Empty Normal Means its vox is empty
	if(voxData.x == 0.0f &&
		voxData.y == 0.0f &&
		voxData.z == 0.0f)
	{
		uint writeIndex = atomicAdd(objectGridInfo[objId].voxCount, 1);
		voxelArrayRender[writeIndex].normal = voxData.xyz;
		voxelArrayRender[writeIndex].color = floatBitsToUint(voxData.w);

		voxelPacked[writeIndex] = PackVoxelData(voxId, objId);
	}
}