#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRenderWorld.vert
	Author		: Bora Yalciner
	Description	:

		Renders World Space voxels
*/

#define IN_CUBE_POS layout(location = 0)
#define IN_VOXEL_POS layout(location = 1)
#define IN_VOXEL_NORM layout(location = 2)
//#define IN_VOXEL_ALBEDO layout(location = 3)

#define OUT_COLOR layout(location = 0)

#define U_RENDER_TYPE layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)

#define LU_VOXEL_GRID_INFO layout(std430, binding = 2)

#define DIFFUSE_ALBEDO 0
#define SPECULAR_ALBEDO 1
#define NORMAL 2

struct GridInfo
{
	vec4 position;		// World Position of the voxel grid, last component is span
	uvec4 dimension;	// Voxel Grid Dimentions, last component is depth of the SVO
};

// Input
in IN_CUBE_POS vec3 vPos;
in IN_VOXEL_POS uint voxPos;
in IN_VOXEL_NORM uint voxRender;
//in IN_VOXEL_ALBEDO vec4 voxAlbedo;

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

LU_VOXEL_GRID_INFO buffer GridInfoBuffer
{
	GridInfo gridInfo[];
};

uvec4 UnpackPosAndCascade(in uint voxPos)
{
	uvec4 vec;
	vec.x = (voxPos & 0x000003FF);
	vec.y = (voxPos & 0x000FFC00) >> 10;
	vec.z = (voxPos & 0x3FF00000) >> 20;
	vec.w  = (voxPos & 0xC0000000) >> 30;
	return vec;
}

vec3 UnpackNormal(in uint voxRender)
{	
	return unpackSnorm4x8(voxRender).xyz;
}

vec4 UnpackAlbedo(in uint voxRender)
{	
	return unpackUnorm4x8(voxRender);
}

void main(void)
{
	uvec4 voxIndex = UnpackPosAndCascade(voxPos);
	vec3 gridPos = gridInfo[voxIndex.w].position.xyz;
	float gridSpan = gridInfo[voxIndex.w].position.w;
	vec3 deltaPos = gridPos + gridSpan * vec3(voxIndex.xyz);

	// Voxels are in world space
	// Need to determine the scale and relative position wrt the grid
	mat4 voxModel =	mat4( gridSpan,		0.0f,		 0.0f,		  0.0f,
						  0.0f,			gridSpan,	 0.0f,		  0.0f,
						  0.0f,			0.0f,		 gridSpan,	  0.0f,
						  deltaPos.x,	deltaPos.y,	 deltaPos.z,  1.0f);
	gl_Position = projection * view * voxModel * vec4(vPos, 1.0f);
	if(renderType == DIFFUSE_ALBEDO)
	{
		// voxRender is albedo
		fColor = UnpackAlbedo(voxRender).rgb;
	}
	else if(renderType == SPECULAR_ALBEDO)
	{
		// voxRender is albedo
		fColor = UnpackAlbedo(voxRender).aaa;
	}
	else if(renderType == NORMAL)
	{
		// voxRender is normal
		//fColor = (UnpackNormal(voxRender) + 1.0f) * 0.5f;
		fColor = vec3(0.0f, 0.0f, ((UnpackNormal(voxRender) + 1.0f) * 0.5f).y);
	}
}