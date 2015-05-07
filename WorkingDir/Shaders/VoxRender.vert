#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRender.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

#define IN_POS layout(location = 0)
#define IN_COLOR layout(location = 1)
#define IN_VOX_POS layout(location = 2)

#define OUT_COLOR layout(location = 0)

#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_FTRANSFORM layout(binding = 0)
#define U_OBJECT layout(std140, binding = 2)
#define U_MTRANSFORM layout(std140, binding = 1)

// Input
in IN_POS vec3 vPos;

in IN_COLOR vec3 voxColor;
in IN_VOX_POS uvec2 voxPos;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_COLOR vec3 fColor;

// Textures

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
	mat4 viewRotation;
};

LU_OBJECT_GRID_INFO buffer GridInfo
{
	float span;
	uint voxCount;
};

U_OBJECT uniform Object
{
	vec3 aabbMin;
	vec3 aabbMax;
};

U_MTRANSFORM uniform ModelTransform
{
	mat4 model;
	mat3 modelRotation;
};

uvec3 UnpackVoxelData(in uvec2 voxPacked)
{
	uvec3 vec;
	vec.x = voxPacked.x >> 0;
	vec.y = voxPacked.x >> 16;
	vec.z = voxPacked.y >> 0;
	return vec;
}

void main(void)
{
	fColor = voxColor;
	mat4 scaleSpan = mat4(span, 0.0f, 0.0f, 0.0f,
						  0.0f, span, 0.0f, 0.0f,
						  0.0f, 0.0f, span, 0.0f,
						  0.0f, 0.0f, 0.0f, 1.0f);
	vec3 pos = aabbMin + (span * vec3(UnpackVoxelData(voxPos))) + vPos.xyz;
	gl_Position = projection * view * model * scaleSpan * vec4(pos, 1.0f);
}