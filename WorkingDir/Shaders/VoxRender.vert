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
#define LU_MTRANSFORM layout(std430, binding = 4) restrict readonly

#define U_FTRANSFORM layout(std140, binding = 0)

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

LU_MTRANSFORM buffer ModelTransform
{
	struct
	{
		mat4 model;
		mat4 modelRotation;
	} modelTransforms[];
};

uvec4 UnpackVoxelDataAndObjId(in uvec4 voxPacked)
{
	uvec4 vec;
	vec.x = (voxPacked.x & 0x000001FF);
	vec.y = (voxPacked.x & 0x0003FE00) >> 9;
	vec.z = (voxPacked.x & 0x07FC0000) >> 18;
	vec.w = (voxPacked.z & 0xFFFF0000) >> 16;
	return vec;
}

void main(void)
{
	fColor = voxColor.rgb;

	uvec4 voxIndex = UnpackVoxelDataAndObjId(voxPos);
	uint objId = voxIndex.w;

	float span = objectGridInfo[objId].span;
	vec3 deltaPos = objectAABBInfo[objId].aabbMin.xyz + 
					(span * vec3(voxIndex.xyz));
	mat4 voxModel =	mat4( span,		0.0f,		0.0f,		0.0f,
						  0.0f,			span,		0.0f,		0.0f,
						  0.0f,			0.0f,		span,		0.0f,
						  deltaPos.x,	deltaPos.y,	deltaPos.z, 1.0f);
	gl_Position = projection * view * modelTransforms[objId].model * voxModel * vec4(vPos, 1.0f);
}