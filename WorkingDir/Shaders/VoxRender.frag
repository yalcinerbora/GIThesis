#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRender.frag
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

// Definitions
#define IN_COLOR layout(location = 0)

// Input
in IN_COLOR vec3 fColor;

// Output
out vec4 frameBuffer;

// Textures

// Textures

// Uniforms

void main(void)
{
	frameBuffer = vec4(fColor, 1.0f);
}