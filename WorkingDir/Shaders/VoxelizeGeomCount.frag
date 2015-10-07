#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeomCount.frag
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict
#define U_OBJ_ID layout(location = 4)

// Uniforms
U_OBJ_ID uniform uint objId;

// Large Uniform Buffers
LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

// Definitions
void main(void)
{
	atomicAdd(objectGridInfo[objId].voxCount, 1);
}