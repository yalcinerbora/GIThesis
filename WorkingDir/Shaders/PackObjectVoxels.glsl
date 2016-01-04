#version 430
				
// Definitions
#define LU_VOXEL_NORM_POS layout(std430, binding = 0) restrict
#define LU_VOXEL_RENDER layout(std430, binding = 1) restrict 
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict readonly
#define LU_VOXEL_IDS layout(std430, binding = 3) restrict
#define LU_INDEX_CHECK layout(std430, binding = 4) restrict

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)
#define U_MAX_CACHE_SIZE layout(location = 5)
#define U_OBJ_TYPE layout(location = 6)
#define U_IS_MIP layout(location = 7)

#define I_VOX_READ layout(rg32ui, binding = 2) restrict

// I-O
U_OBJ_TYPE uniform uint objType;
U_OBJ_ID uniform uint objId;
U_IS_MIP uniform uint isMip;
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_MAX_CACHE_SIZE uniform uint maxSize;

LU_INDEX_CHECK buffer CountArray
{
	uint writeIndex;
};

LU_VOXEL_NORM_POS buffer VoxelNormPosArray
{
	uvec2 voxelNormPos[];
};

LU_VOXEL_IDS buffer VoxelIdsArray
{
	uvec2 voxelIds[];
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

uvec2 PackVoxelNormPos(in uvec3 voxCoord,
					   in uint normal,
					   in uint isMip)
{
	uvec2 result = uvec2(0);
	
	// Voxel Ids 9 Bit Each (last 5 bit is span depth)
	unsigned int value = 0;
	value |= isMip << 30;
	value |= voxCoord.z << 20;
	value |= voxCoord.y << 10;
	value |= voxCoord.x;
	result.x = value;

	// Normal is Already Packed (XYZ8 SNROM format)
	result.y = normal;
	return result;
}

uvec2 PackVoxelIds(in uint objId,
				   in uint objType,
				   in uint renderDataLoc)
{
	uvec2 result = uvec2(0);

	// Object Id (13 bit batch id, 16 bit object id)
	// Last 2 bits is for object type
	unsigned int value = 0;
	value |= objType << 30;
	value |= 0 << 16;
	value |= objId;
	result.x = value;

	result.y = renderDataLoc;
	return result;
}

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main(void)
{
	uvec3 voxId = gl_GlobalInvocationID.xyz;
	if(voxId.x < (voxDim.x)  &&
		voxId.y < (voxDim.y) &&
		voxId.z < (voxDim.z))
	{
		uvec2 voxData = imageLoad(voxelData, ivec3(voxId)).xy;

		// Empty Normal Means its vox is empty
		if(voxData.x != 0xFFFFFFFF ||
			voxData.y != 0xFFFFFFFF)
		{
			uint index = atomicAdd(writeIndex, 1);
			if(index <= maxSize)
			{
				voxelArrayRender[index].color = voxData.y;
				voxelNormPos[index] = PackVoxelNormPos(voxId, voxData.x, isMip);
				voxelIds[index] = PackVoxelIds(objId, objType, index);
			}
			// Reset Color For next iteration
			imageStore(voxelData, ivec3(voxId), uvec4(0xFFFFFFFF));
		}
	}
}