#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)

#define LU_AABB layout(std430, binding = 3) restrict readonly
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict readonly
#define LU_VOXEL_NORM_POS layout(std430, binding = 0) restrict
#define LU_VOXEL_RENDER layout(std430, binding = 1) restrict
#define LU_VOXEL_IDS layout(std430, binding = 4) restrict
#define LU_ATOMIC_COUNTER layout(std430, binding = 9) restrict
#define LU_SORT_INDICES layout(std430, binding = 10) restrict

#define T_COLOR layout(binding = 0)

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)
#define U_OBJ_TYPE layout(location = 6)
#define U_SPAN_RATIO layout(location = 7)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
in IN_POS vec3 fPos;

// Output

// Textures
uniform T_COLOR sampler2D colorTex;

// Uniforms
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_OBJ_TYPE uniform uint objType;
U_OBJ_ID uniform uint objId;
U_SPAN_RATIO uniform uint spanRatio;

LU_AABB buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

LU_SORT_INDICES buffer SortIndices
{
	unsigned int sortIndex[];
};

LU_VOXEL_NORM_POS buffer VoxelNormPos
{
	uvec2 voxelNormPos[];
};

LU_VOXEL_IDS buffer VoxelIds
{
	uvec2 voxelIds[];
};

LU_VOXEL_RENDER buffer VoxelRender
{
	uint voxelRenderData[];
};

LU_ATOMIC_COUNTER buffer AtomicCounter
{
	unsigned int allocation;
};

uint PackColor(vec3 color) 
{
	uint result;
	color *= vec3(255.0f);
	result = uint(/*color.a*/0) << 24;
	result |= uint(color.b) << 16;
	result |= uint(color.g) << 8;
	result |= uint(color.r);

    return result;
}

uvec2 PackVoxelNormPos(in uvec3 voxCoord,
					   in vec3 normal,
					   in uint spanDepth)
{
	uvec2 result = uvec2(0);
	
	// Voxel Ids 9 Bit Each (last 5 bit is span depth)
	unsigned int value = 0;
	value |= spanDepth << 27;
	value |= voxCoord.z << 18;
	value |= voxCoord.y << 9;
	value |= voxCoord.x;
	result.x = value;

	// 1615 XY Format
	// 32 bit format LS 16 bits are X
	// MSB is the sign of Z
	// Rest is Y
	// both x and y is SNORM types
	uvec2 value2 = uvec2(0.0f);
	value2.x = uint((normal.x * 0.5f + 0.5f) * 0xFFFF);
	value2.y = uint((normal.y * 0.5f + 0.5f) * 0x7FFF);
	value2.y |= (floatBitsToUint(normal.z) >> 16) & 0x00008000;
	result.y = value2.y << 16;
	result.y |= value2.x;

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

void main(void)
{
	// Data Packing forming
	vec3 color = texture2D(colorTex, fUV).rgb;

	// Because of the MSAA voxel count may be slightly different
	// Omit all exceeding voxels
	uint index = atomicAdd(allocation, 1);
	if(index >= objectGridInfo[objId].voxCount) return;

	// Calculate Position wrt voxel
	uvec3 voxelCoord = uvec3(floor((fPos - objectAABBInfo[objId].aabbMin.xyz) / objectGridInfo[objId].span));

	// Store
	if(voxelCoord.x < voxDim.x &&
		voxelCoord.y < voxDim.y &&
		voxelCoord.z < voxDim.z)
	{
		voxelNormPos[index] = PackVoxelNormPos(voxelCoord, fNormal, spanRatio);
		voxelIds[index] = PackVoxelIds(objId, objType, 0);
		voxelRenderData[index] = PackColor(color);
		sortIndex[index] = index;
	}
}