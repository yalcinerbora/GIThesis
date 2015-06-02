#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRender.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

#define IN_POS layout(location = 0)
#define IN_VOX_COLOR layout(location = 1)
#define IN_VOX_POS layout(location = 2)
#define IN_VOX_NORMAL layout(location = 3)

#define OUT_COLOR layout(location = 0)

#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict
#define LU_AABB layout(std430, binding = 3) restrict readonly

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_MTRANSFORM layout(std140, binding = 1)

// Input
in IN_POS vec3 vPos;

in IN_VOX_COLOR vec4 voxColor;
in IN_VOX_POS uvec2 voxPos;
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

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

LU_AABB buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};

U_MTRANSFORM uniform ModelTransform
{
	mat4 model;
	mat3 modelRotation;
};

uvec4 UnpackVoxelData(in uvec2 voxPacked)
{
	uvec4 vec;
	vec.x = voxPacked.x & 0x0000FFFF;
	vec.y = voxPacked.x >> 16;
	vec.z = voxPacked.y & 0x0000FFFF;
	vec.w = voxPacked.y >> 16;
	return vec;
}

void main(void)
{
	fColor = voxColor.rgb;

	uvec4 voxIndex = UnpackVoxelData(voxPos);
	uint objId = voxIndex.w;
	float span = objectGridInfo[objId].span;
	vec3 deltaPos = objectAABBInfo[objId].aabbMin.xyz + 
					(span * vec3(voxIndex.xyz)) +
					vec3(span * 0.5f);
	mat4 voxModel =	mat4( span,		0.0f,		0.0f,		0.0f,
						  0.0f,			span,		0.0f,		0.0f,
						  0.0f,			0.0f,		span,		0.0f,
						  deltaPos.x,	deltaPos.y,	deltaPos.z, 1.0f);
	gl_Position = projection * view * model * voxModel * vec4(vPos, 1.0f);
}