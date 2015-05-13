#version 430
				
// Definitions
#define LU_VOXEL layout(std430, binding = 0) coherent 
#define LU_VOXEL_RENDER layout(std430, binding = 1) coherent 
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) coherent readonly
#define LU_INDEX_CHECK layout(std430, binding = 4) coherent

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)
#define U_MAX_CACHE_SIZE layout(location = 5)

#define I_VOX_READ layout(rgba32f, binding = 2) coherent

// I-O
U_OBJ_ID uniform uint objId;
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_MAX_CACHE_SIZE uniform uint maxSize;

LU_INDEX_CHECK buffer CountArray
{
	uint writeIndex;
};

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
	uvec2 vec = uvec2(0);
	vec.x = voxCoord.x;
	vec.x |= voxCoord.y << 16;
	vec.y = voxCoord.z;
	vec.y |= objId << 16;
	return vec;
}
		
layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main(void)
{
	uvec3 voxId = gl_GlobalInvocationID.xyz;
	if(voxId.x <= voxDim.x &&
		voxId.y <= voxDim.y &&
		voxId.z <= voxDim.z)
	{
		//memoryBarrier();
		vec4 voxData = imageLoad(voxelData, ivec3(voxId));

		// Empty Normal Means its vox is empty
		if(voxData.x != 0.0f ||
			voxData.y != 0.0f ||
			voxData.z != 0.0f)
		{
			uint index = atomicAdd(writeIndex, 1);
			if(index <= maxSize)
			{
				voxelArrayRender[index].normal = voxData.xyz;
				voxelArrayRender[index].color = floatBitsToUint(voxData.w);
				voxelPacked[index] = PackVoxelData(voxId, objId);
			}
		}
	}
	// Reset Color For next iteration
	imageStore(voxelData, ivec3(voxId), vec4(0.0f));
	//memoryBarrier();
}