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
#define IN_VOX_NORM_POS layout(location = 2)
#define IN_VOX_IDS layout(location = 3)
#define IN_VOX_WEIGHT layout(location = 4)

#define OUT_COLOR layout(location = 0)

#define U_RENDER_TYPE layout(location = 0)
#define U_SPAN layout(location = 1)

#define LU_AABB layout(std430, binding = 3) restrict readonly
#define LU_MTRANSFORM layout(std430, binding = 4) restrict readonly
#define LU_MTRANSFORM_INDEX layout(std430, binding = 5) restrict readonly
#define LU_JOINT_TRANS layout(std430, binding = 6) restrict readonly

#define U_FTRANSFORM layout(std140, binding = 0)

#define RENDER_TYPE_COLOR 0
#define RENDER_TYPE_NORMAL 1

// Input
in IN_POS vec3 vPos;
in IN_VOX_COLOR vec4 voxColor;
in IN_VOX_NORM_POS uvec2 voxNormPos;
in IN_VOX_IDS uvec2 voxIds;
in IN_VOX_WEIGHT uvec2 voxWeights;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_COLOR vec3 fColor;

// Textures

// Uniforms
U_RENDER_TYPE uniform uint renderType;
U_SPAN uniform float span;

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
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

LU_MTRANSFORM_INDEX buffer ModelTransformID
{
	uint modelTransformIds[];
};

LU_JOINT_TRANS buffer JointTransforms
{
	struct
	{
		mat4 final;
		mat4 finalRot;
	} jointTransforms[];
};


vec3 UnpackNormal(in uint voxNormPosY)
{	
	return unpackSnorm4x8(voxNormPosY).xyz;
}

uvec4 UnpackVoxelDataAndObjId(in uint voxNormPosX, in uint voxIdsX)
{
	uvec4 vec;
	vec.x = (voxNormPosX & 0x000003FF);
	vec.y = (voxNormPosX & 0x000FFC00) >> 10;
	vec.z = (voxNormPosX & 0x3FF00000) >> 20;
	vec.w = (voxIdsX & 0x0000FFFF);
	return vec;
}

uvec4 UnpackVoxelIndices(uint wIPacked)
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
	uvec4 voxIndex = UnpackVoxelDataAndObjId(voxNormPos.x, voxIds.x);
	uint objId = voxIndex.w;
	uint transformId = modelTransformIds[objId];

	uvec4 vWIndex = UnpackVoxelIndices(voxWeights.y);
	vec4 vWeight = UnpackVoxelWeights(voxWeights.x);

	vec3 deltaPos = objectAABBInfo[objId].aabbMin.xyz + (span * vec3(voxIndex.xyz));
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

	if(renderType == RENDER_TYPE_COLOR)
		fColor = voxColor.rgb;
	else if(renderType == RENDER_TYPE_NORMAL)
	{
		vec3 normalModel = UnpackNormal(voxNormPos.y);
		vec3 norm = vec3(0.0f);	
		norm += (mat3(jointTransforms[vWIndex[0]].finalRot) * normalModel) * vWeight[0];
		norm += (mat3(jointTransforms[vWIndex[1]].finalRot) * normalModel) * vWeight[1];
		norm += (mat3(jointTransforms[vWIndex[2]].finalRot) * normalModel) * vWeight[2];
		norm += (mat3(jointTransforms[vWIndex[3]].finalRot) * normalModel) * vWeight[3];

		norm = mat3(modelTransforms[transformId].modelRotation) * norm;

		fColor = norm;
	}

}