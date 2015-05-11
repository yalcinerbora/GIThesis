#version 430
				
// Definitions
#define LU_VOXEL layout(std430, binding = 0)
#define LU_VOXEL_RENDER layout(std430, binding = 1)
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)

#define I_VOX_READ layout(rgba32f, binding = 2) restrict readonly

#define MAX_GRID_DIM 128.0f
#define INCREMENT_FACTOR 0.2f

// I-O
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_OBJ_ID uniform uint objId;

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
	voxId.xyz  = uvec3(gl_LocalInvocationID.xy * (gl_WorkGroupID.x % 16), 
						gl_WorkGroupID.x / 16);

	if(voxId.x >= voxDim.x || 
		voxId.y >= voxDim.y) return;

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