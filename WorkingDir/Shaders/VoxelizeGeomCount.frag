#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Determines Count of the vox geom
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)

#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;

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