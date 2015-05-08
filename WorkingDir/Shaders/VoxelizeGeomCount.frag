#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Determines Count of the vox geom
*/

// Definitions
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)
#define U_TOTAL_OBJ_COUNT layout(location = 3)

// Input

// Output
out vec4 colorDebug;

// Textures

// Textures

// Uniforms
U_TOTAL_OBJ_COUNT uniform uint index;

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

void main(void)
{
	atomicAdd(objectGridInfo[index].voxCount, 1);
	colorDebug = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}