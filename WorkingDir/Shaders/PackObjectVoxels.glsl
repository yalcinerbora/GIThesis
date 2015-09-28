#version 430
				
// Definitions
#define LU_VOXEL layout(std430, binding = 0) restrict 
#define LU_VOXEL_RENDER layout(std430, binding = 1) restrict 
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict readonly
#define LU_INDEX_CHECK layout(std430, binding = 4) restrict

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)
#define U_OBJ_TYPE layout(location = 6)
#define U_SPAN_RATIO layout(location = 7)
#define U_MAX_CACHE_SIZE layout(location = 5)

#define I_VOX_READ layout(rgba16ui, binding = 2) restrict

// I-O
U_OBJ_TYPE uniform uint objType;
U_OBJ_ID uniform uint objId;
U_SPAN_RATIO uniform uint spanRatio;
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_MAX_CACHE_SIZE uniform uint maxSize;

LU_INDEX_CHECK buffer CountArray
{
	uint writeIndex;
};

LU_VOXEL buffer VoxelArray
{
	uvec4 voxelPacked[];
};

LU_VOXEL_RENDER buffer VoxelArrayRender
{
	struct
	{
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

uniform I_VOX_READ uimage3D voxelData;

uint MergeColor(uvec2 colorShort2)
{
	uint result;
	result = colorShort2.y << 16;
	result |= colorShort2.x;
	return result;
}

uvec4 PackVoxelData(in uvec3 voxCoord,
					in uvec2 normal,
					in uint objId,
					in uint objType,
					in uint renderDataLoc)
{
	uvec4 result = uvec4(0);
	
	// Here Pack the voxels
	unsigned int value = 0;
	value |= spanRatio << 27;
	value |= voxCoord.z << 18;
	value |= voxCoord.y << 9;
	value |= voxCoord.x;
	result.x = value;

	value = 0;
	value |= normal.y << 16;
	value |= normal.x;
	
	result.y = value;

	value = 0;
	value |= objType << 30;
	value |= 0 << 16;
	value |= objId;
	result.z = value;

	result.w = renderDataLoc;
	return result;
}
		
layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main(void)
{
	uvec3 voxId = gl_GlobalInvocationID.xyz;
	if(voxId.x < voxDim.x &&
		voxId.y < voxDim.y &&
		voxId.z < voxDim.z)
	{
		//memoryBarrier();
		uvec4 voxData = imageLoad(voxelData, ivec3(voxId));

		// Empty Normal Means its vox is empty
		if(voxData.x != 0xFFFF ||
			voxData.y != 0xFFFF)
		{
			uint index = atomicAdd(writeIndex, 1);
			if(index <= maxSize)
			{
				voxelArrayRender[index].color = MergeColor(voxData.zw);
				voxelPacked[index] = PackVoxelData(voxId, voxData.xy, objId, objType, index);
			}
		}
	}
	// Reset Color For next iteration
	imageStore(voxelData, ivec3(voxId), uvec4(0xFFFF));
	//memoryBarrier();
}