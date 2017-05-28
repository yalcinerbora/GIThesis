#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRenderWorld.vert
	Author		: Bora Yalciner
	Description	:

		Renders World Space voxels
*/

#define IN_POS layout(location = 0)
#define IN_VOX_COLOR layout(location = 1)
#define IN_VOX_NORM_POS layout(location = 2)

#define OUT_COLOR layout(location = 0)

#define U_RENDER_TYPE layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_VOXEL_GRID_INFO layout(std140, binding = 2)

#define RENDER_TYPE_COLOR 0
#define RENDER_TYPE_NORMAL 1

// Input
in IN_POS vec3 vPos;
in IN_VOX_COLOR vec4 voxColor;
in IN_VOX_NORM_POS uvec2 voxNormPos;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_COLOR vec3 fColor;

// Textures

// Uniforms
U_RENDER_TYPE uniform uint renderType;

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

U_VOXEL_GRID_INFO uniform GridInfo
{
	vec4 position;		// World Position of the voxel grid, last component is span
	uvec4 dimension;	// Voxel Grid Dimentions, last component is depth of the SVO
};

vec3 UnpackNormal(in uint voxNormPosY)
{	
	return unpackSnorm4x8(voxNormPosY).xyz;
}

uvec4 UnpackVoxelDataAndSpan(in uint voxNormPosX)
{
	uvec4 vec;
	vec.x = (voxNormPosX & 0x000003FF);
	vec.y = (voxNormPosX & 0x000FFC00) >> 10;
	vec.z = (voxNormPosX & 0x3FF00000) >> 20;
	vec.w  = (voxNormPosX & 0xF8000000) >> 27;
	return vec;
}

void main(void)
{
	// Color directly to fragment
	if(renderType == RENDER_TYPE_COLOR)
		fColor = voxColor.rgb;
	else if(renderType == RENDER_TYPE_NORMAL)
		fColor = UnpackNormal(voxNormPos.y);

	// Voxels are in world space
	// Need to determine the scale and relative position wrt the grid
	uvec4 voxIndex = UnpackVoxelDataAndSpan(voxNormPos.x);
	float span = position.w;
	vec3 deltaPos = position.xyz + position.w * vec3(voxIndex.xyz);
	mat4 voxModel =	mat4( span,			0.0f,		0.0f,		0.0f,
						  0.0f,			span,		0.0f,		0.0f,
						  0.0f,			0.0f,		span,		0.0f,
						  deltaPos.x,	deltaPos.y,	deltaPos.z, 1.0f);
	gl_Position = projection * view * voxModel * vec4(vPos, 1.0f);
}