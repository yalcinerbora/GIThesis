#version 430
/*	
	**Voxelize Count Shader**
	
	File Name	: VoxelizeGeomCount.frag
	Author		: Bora Yalciner
	Description	:

		Counts How Many Voxels this Object Can Generate
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)

#define LU_AABB layout(std430, binding = 3) readonly
#define LU_OBJECT_VOXEL_INFO layout(std430, binding = 2) writeonly
#define LU_TOTAL_VOX_COUNT layout(std430, binding = 4) writeonly

#define I_LOCK layout(r32ui, binding = 0) coherent volatile

#define U_SPAN layout(location = 1)
#define U_SEGMENT_SIZE layout(location = 2)
#define U_OBJ_ID layout(location = 4)
#define U_SPLIT_CURRENT layout(location = 7)
#define U_TEX_SIZE layout(location = 8)

// Input
in IN_POS vec3 fPos;

// Images
uniform I_LOCK uimage3D lock;

// Uniform Constants
U_SPAN uniform float span;
U_SEGMENT_SIZE uniform float segmentSize;
U_SPLIT_CURRENT uniform uvec3 currentSplit;
U_OBJ_ID uniform uint objId;
U_TEX_SIZE uniform uint texSize3D;

// Shader Torage
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
		uint voxCount;
		uint voxOffset;
	} voxInfo[];
};

LU_TOTAL_VOX_COUNT buffer TotalVox
{
	uint totalVox;
};

void main(void)
{
	// interpolated object space pos
	vec3 aabbMin = objectAABBInfo[objId].aabbMin.xyz;
	aabbMin += vec3(currentSplit) * vec3(segmentSize);
	vec3 voxelCoord = floor((fPos - aabbMin) / span);
	
	ivec3 iCoord = ivec3(voxelCoord);
	if(iCoord.x < texSize3D &&
	   iCoord.y < texSize3D &&
	   iCoord.z < texSize3D &&
	   iCoord.x >= 0 &&
	   iCoord.y >= 0 &&
	   iCoord.z >= 0)
	{
		if(imageAtomicExchange(lock, iCoord, 1) == 0) 
		{
			atomicAdd(voxInfo[objId].voxCount, 1);
			atomicAdd(totalVox, 1);
		}
	}
}