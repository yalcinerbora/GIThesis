#version 430
/*	
	**Render Voxel**
	
	File Name	: VoxRender.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

#define IN_CUBE_POS layout(location = 0)
#define IN_VOXEL_POS layout(location = 1)
#define IN_VOXEL_NORM layout(location = 2)
#define IN_VOXEL_ALBEDO layout(location = 3)
#define IN_VOXEL_WEIGHT layout(location = 4)

#define OUT_COLOR layout(location = 0)

#define U_RENDER_TYPE layout(location = 0)
#define U_SPAN layout(location = 1)
#define U_DRAW_ID layout(location = 2)

#define LU_AABB layout(std430, binding = 3) restrict readonly
#define LU_MTRANSFORM layout(std430, binding = 4) restrict readonly
#define LU_MTRANSFORM_INDEX layout(std430, binding = 5) restrict readonly
#define LU_JOINT_TRANS layout(std430, binding = 6) restrict readonly

#define U_FTRANSFORM layout(std140, binding = 0)

#define DIFFUSE_ALBEDO 0
#define SPECULAR_ALBEDO 1
#define NORMAL 2

struct AABB
{
	vec4 aabbMin;
	vec4 aabbMax;
};

struct ModelTransform
{
	mat4 model;
	mat4 modelRotation;
};

struct JointTransform
{
	mat4 final;
	mat4 finalRot;
};

// Input
in IN_CUBE_POS vec3 vPos;
in IN_VOXEL_POS uint voxPos;
in IN_VOXEL_NORM uint voxNorm;
in IN_VOXEL_ALBEDO vec4 voxAlbedo;
in IN_VOXEL_WEIGHT uvec2 voxWeights;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_COLOR vec3 fColor;

// Textures

// Uniforms
U_RENDER_TYPE uniform uint renderType;
U_SPAN uniform float span;
U_DRAW_ID uniform uint drawId;

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

LU_AABB buffer AABBBuffer
{
	AABB objectAABBInfo[];
};

LU_MTRANSFORM buffer ModelTransformBuffer
{
	ModelTransform modelTransforms[];
};

LU_MTRANSFORM_INDEX buffer ModelTransformID
{
	uint modelTransformIds[];
};

LU_JOINT_TRANS buffer JointTransformBuffer
{
	JointTransform jointTransforms[];
};

uvec3 UnpackPos(in uint voxPos)
{
	uvec3 vec;
	vec.x = (voxPos & 0x000003FF);
	vec.y = (voxPos & 0x000FFC00) >> 10;
	vec.z = (voxPos & 0x3FF00000) >> 20;
	return vec;
}

vec3 UnpackNormal(in uint voxNorm)
{	
	return unpackSnorm4x8(voxNorm).xyz;
}

uvec4 UnpackVoxelWeightIndices(uint wIPacked)
{
	uvec4 result;
	result.x = (wIPacked >> 0) & 0x000000FF;
	result.y = (wIPacked >> 8) & 0x000000FF;
	result.z = (wIPacked >> 16) & 0x000000FF;
	result.w = (wIPacked >> 24) & 0x000000FF;
	return result;
}

vec4 UnpackVoxelWeights(uint wPacked)
{
	return unpackUnorm4x8(wPacked);
}

void main(void)
{
	uvec3 voxIndex = UnpackPos(voxPos);
	uint transformId = modelTransformIds[drawId];

	uvec4 vWIndex = UnpackVoxelWeightIndices(voxWeights.y);
	vec4 vWeight = UnpackVoxelWeights(voxWeights.x);

	vec3 deltaPos = objectAABBInfo[drawId].aabbMin.xyz + (span * vec3(voxIndex));
	mat4 voxModel =	mat4( span,			0.0f,		0.0f,		0.0f,
						  0.0f,			span,		0.0f,		0.0f,
						  0.0f,			0.0f,		span,		0.0f,
						  deltaPos.x,	deltaPos.y,	deltaPos.z, 1.0f);

	// Animations
	vec4 pos = vec4(0.0f);
	pos += jointTransforms[vWIndex[0]].final * voxModel * vec4(vPos, 1.0f) * vWeight[0];
	pos += jointTransforms[vWIndex[1]].final * voxModel * vec4(vPos, 1.0f) * vWeight[1];
	pos += jointTransforms[vWIndex[2]].final * voxModel * vec4(vPos, 1.0f) * vWeight[2];
	pos += jointTransforms[vWIndex[3]].final * voxModel * vec4(vPos, 1.0f) * vWeight[3];
	pos = modelTransforms[transformId].model * pos;
	gl_Position = projection * view * pos;

	if(renderType == DIFFUSE_ALBEDO)
	{
		fColor = voxAlbedo.rgb;
	}
	else if(renderType == SPECULAR_ALBEDO)
	{
		fColor = voxAlbedo.aaa;
	}
	else if(renderType == NORMAL)
	{
		vec3 normalModel = UnpackNormal(voxNorm);
		vec3 norm = vec3(0.0f);	
		norm += (mat3(jointTransforms[vWIndex[0]].finalRot) * normalModel) * vWeight[0];
		norm += (mat3(jointTransforms[vWIndex[1]].finalRot) * normalModel) * vWeight[1];
		norm += (mat3(jointTransforms[vWIndex[2]].finalRot) * normalModel) * vWeight[2];
		norm += (mat3(jointTransforms[vWIndex[3]].finalRot) * normalModel) * vWeight[3];

		norm = mat3(modelTransforms[transformId].modelRotation) * norm;
		fColor = norm;
	}
}