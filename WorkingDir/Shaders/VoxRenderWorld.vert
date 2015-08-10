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
#define IN_VOX_POS layout(location = 2)
#define IN_VOX_NORMAL layout(location = 3)

#define OUT_COLOR layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_VOXEL_GRID_INFO layout(std140, binding = 2)

// Input
in IN_POS vec3 vPos;
in IN_VOX_COLOR vec4 voxColor;
in IN_VOX_POS uvec4 voxPos;
in IN_VOX_NORMAL vec3 voxNormal;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_COLOR vec3 fColor;

// Textures

// Uniforms
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

uvec4 UnpackVoxelData(in uvec4 voxPacked)
{
	uvec4 vec;
	vec.x = (voxPacked.x & 0x000003FF);
	vec.y = (voxPacked.x & 0x000FFC00) >> 10;
	vec.z = (voxPacked.x & 0x3FF00000) >> 20;
	vec.w = (voxPacked.z & 0xFFFF0000) >> 16;
	return vec;
}

void main(void)
{
	fColor = voxColor.rgb;

	uvec4 voxIndex = UnpackVoxelData(voxPos);
	uint objId = voxIndex.w;
	float span = position.w;
	vec3 deltaPos = position.xyz + (span * vec3(voxIndex.xyz)) + vec3(span * 0.5f);
	mat4 voxModel =	mat4( span,			0.0f,		0.0f,		0.0f,
						  0.0f,			span,		0.0f,		0.0f,
						  0.0f,			0.0f,		span,		0.0f,
						  deltaPos.x,	deltaPos.y,	deltaPos.z, 1.0f);
	gl_Position = projection * view * voxModel * vec4(vPos, 1.0f);
}