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

// Input

// Output

// Textures

// Textures

// Uniforms
LU_OBJECT_GRID_INFO buffer GridInfo
{
	float span;
	uint voxCount;
};

void main(void)
{
	uint location = atomicAdd(voxCount, 1);
}