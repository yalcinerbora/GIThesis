#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRender.frag
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry with cascade culling
*/

// Definitions
#define IN_COLOR layout(location = 0)
#define IN_CULL layout(location = 1)

// Input
in IN_COLOR vec3 fColor;
flat in IN_CULL int fCull;

// Output
out vec4 frameBuffer;

// Textures

// Textures

// Uniforms

void main(void)
{
	if(fCull == 1) discard;
	frameBuffer = vec4(fColor, 1.0f);
}