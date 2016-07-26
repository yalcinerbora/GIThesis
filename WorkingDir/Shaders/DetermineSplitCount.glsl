#version 430
/*	
	**Determine Split Count Shader**
	
	File Name	: DetermineSplitCount.frag
	Author		: Bora Yalciner
	Description	: 

		Determine how many splits we need to make in order to cover this
		object with this span size
*/
				
// Definitions
#define LU_AABB layout(std430, binding = 3) readonly
#define LU_OBJECT_SPLIT_INFO layout(std430, binding = 4) restrict writeonly
#define LU_OBJECT_VOXEL_INFO layout(std430, binding = 2) restrict writeonly

#define U_TOTAL_OBJ_COUNT layout(location = 0)
#define U_SPAN layout(location = 1)
#define U_GRID_DIM layout(location = 2)
#define U_VOX_LIMIT layout(location = 3)

#define BLOCK_SIZE 256

U_TOTAL_OBJ_COUNT uniform uint objCount;
U_SPAN uniform float span;
U_GRID_DIM uniform uint gridDim;
U_VOX_LIMIT uniform uint voxLimit;

LU_OBJECT_SPLIT_INFO buffer SplitInfo
{
	uvec2 splitInfo[];
};

LU_AABB buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};

LU_OBJECT_VOXEL_INFO buffer VoxelInfo
{
	struct
	{
		float span;
		uint voxCount;
	} voxInfo[];
};

uvec2 PackUInt4x16(uvec4 data)
{
	uvec2 result = uvec2(0x00000000);
	result.x |= (data.x & 0x0000FFFF) << 0;
	result.x |= (data.y & 0x0000FFFF) << 16;

	result.y |= (data.z & 0x0000FFFF) << 0;
	result.y |= (data.w & 0x0000FFFF) << 16;
	return result;
}

layout (local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
void main(void)
{
	uint globalId = gl_GlobalInvocationID.x;
	if(globalId >= objCount) return;

	float cameraCoverage = float(gridDim) * span;
	vec3 objDim = objectAABBInfo[globalId].aabbMax.xyz - 
				  objectAABBInfo[globalId].aabbMin.xyz;
	vec3 noOfSamples = objDim / cameraCoverage;
	
	uvec4 splits = max(uvec4(ceil(noOfSamples), 0.0f), 1);
	uint maxSplits = voxLimit / gridDim;
	float objSpan = span;
	if(any(greaterThan(splits, uvec4(maxSplits))))
	{
		splits = uvec4(0);
		objSpan = 0.0f;
	}
	splitInfo[globalId] = PackUInt4x16(splits);
	voxInfo[globalId].span = objSpan;
	voxInfo[globalId].voxCount = 0;
}