#version 430
				
// Definitions
#define LU_VOXEL_NORM_POS layout(std430, binding = 0) restrict
#define LU_VOXEL_COLOR layout(std430, binding = 1) restrict 
#define LU_VOXEL_WEIGHT layout(std430, binding = 9) restrict
#define LU_INDEX_CHECK layout(std430, binding = 4) restrict
#define LU_VOXEL_IDS layout(std430, binding = 5) restrict
#define LU_NORMAL_SPARSE layout(std430, binding = 6) restrict readonly
#define LU_COLOR_SPARSE layout(std430, binding = 7) restrict readonly
#define LU_WEIGHT_SPARSE layout(std430, binding = 8) restrict readonly

#define I_LOCK layout(r32ui, binding = 0) restrict readonly

#define U_OBJ_ID layout(location = 4)
#define U_MAX_CACHE_SIZE layout(location = 5)
#define U_OBJ_TYPE layout(location = 6)
#define U_SPLIT_CURRENT layout(location = 7)
#define U_TEX_SIZE layout(location = 8)
#define U_IS_MIP layout(location = 9)

// I-O
U_OBJ_TYPE uniform uint objType;
U_OBJ_ID uniform uint objId;
U_IS_MIP uniform uint isMip;
U_MAX_CACHE_SIZE uniform uint maxSize;
U_SPLIT_CURRENT uniform uvec3 currentSplit;
U_TEX_SIZE uniform uint texSize3D;

uniform I_LOCK uimage3D lock;

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

LU_VOXEL_COLOR buffer VoxelColorArray
{
	uint voxColor[];
};

LU_VOXEL_WEIGHT buffer VoxelWeightArray
{
	uvec2 weights[];
};

LU_COLOR_SPARSE buffer ColorBuffer 
{
	vec4 colorSparse[];
};

LU_NORMAL_SPARSE buffer NormalBuffer 
{
	vec4 normalSparse[];
};

LU_WEIGHT_SPARSE buffer WeightBuffer 
{
	uvec2 weightSparse[];
};

uvec2 PackVoxelNormPos(in uvec3 voxCoord,
					   in vec4 normal,
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

	// Normal packed with XYZ8 SNROM format (W is empty and will be used as volume info)
	result.y = packSnorm4x8(normal);
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
	if(voxId.x < texSize3D &&
	   voxId.y < texSize3D &&
	   voxId.z < texSize3D)
	{
		ivec3 iCoord = ivec3(voxId);
		uint coord = iCoord.z * texSize3D * texSize3D +
					 iCoord.y * texSize3D +
					 iCoord.x;
		// DEBUG
		//uint voxOccupation = imageLoad(lock, ivec3(voxId)).x;
		//if(voxOccupation == 1) atomicAdd(writeIndex, 1);
		
		vec4 normal = normalSparse[coord];
		vec4 color = colorSparse[coord];
		uvec2 weight = weightSparse[coord];

		// Empty Normal Means its vox is empty
		if(normal.w > 0.0f)
		{
			// Average Divide
			color.xyz /= normal.w;
			normal.xyz /= normal.w;
			normal.w = 0.0f;

			uint index = atomicAdd(writeIndex, 1);
			if(index <= maxSize)
			{
				voxId += currentSplit * texSize3D;

				voxColor[index] = packUnorm4x8(color);
				voxelNormPos[index] = PackVoxelNormPos(voxId, normal, isMip);
				voxelIds[index] = PackVoxelIds(objId, objType, index);
				weights[index] = weight;
			}
		}
	}
}