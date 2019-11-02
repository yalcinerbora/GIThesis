#version 430
/*	
	**Voxelize Shader Skeletal**
	
	File Name	: VoxelizeGeomSkeletal.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Skeletal Geometry
*/

#define IN_POS layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_UV layout(location = 2)
#define IN_TRANS_INDEX layout(location = 3)
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)
#define OUT_POS layout(location = 2)

#define U_VOLUME_SIZE layout(location = 2)
#define U_VOLUME_CORNER layout(location = 3)

#define LU_MTRANSFORM layout(std430, binding = 4)
#define LU_JOINT_TRANS layout(std430, binding = 6)

#define JOINT_PER_VERTEX 4

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
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_POS vec3 vPos;
in IN_WEIGHT vec4 vWeight;
in IN_WEIGHT_INDEX uvec4 vWIndex;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_NORMAL vec3 fNormal;
out OUT_POS vec3 fPos;
invariant gl_Position;

// Textures

// Uniforms
U_VOLUME_SIZE uniform vec3 volumeSize;
U_VOLUME_CORNER uniform vec3 gridCorner;

LU_MTRANSFORM buffer ModelTransformBuffer
{
	ModelTransform modelTransforms[];
};

LU_JOINT_TRANS buffer JointTransformBuffer
{
	JointTransform jointTransforms[];
};

mat4 orthoFromAABB()
{
	float nearPlane = -volumeSize.z * 0.5f;
	float farPlane = volumeSize.z * 0.5f;
	vec3 gridCenter = gridCorner + volumeSize * 0.5f;

	//	orto	0		0		0
	//	0		orto	0		0
	//	0		0		orto	0
	//	0		0		orto	1
	float zt = nearPlane / (nearPlane - farPlane);
	mat4 ortho = mat4(2.0f / volumeSize.x,	0.0f,					0.0f,							0.0f,
					  0.0f,					2.0f / volumeSize.y,	0.0f,							0.0f,
					  0.0f,					0.0f,					1.0f / (nearPlane - farPlane),	0.0f,
					  0.0f,					0.0f,					zt,								1.0f);
	mat4 view = mat4(1.0f,					0.0f,					0.0f,							0.0f,
					 0.0f,					1.0f,					0.0f,							0.0f,
					 0.0f,					0.0f,					1.0f,							0.0f,
					 gridCenter.z,			gridCenter.z,			gridCenter.z,					1.0f);
	return ortho * view;
}

void main(void)
{

	// Animations
	vec4 pos = vec4(0.0f);
	pos += (jointTransforms[vWIndex[0]].final * vec4(vPos, 1.0f)) * vWeight[0];
	pos += (jointTransforms[vWIndex[1]].final * vec4(vPos, 1.0f)) * vWeight[1];
	pos += (jointTransforms[vWIndex[2]].final * vec4(vPos, 1.0f)) * vWeight[2];
	pos += (jointTransforms[vWIndex[3]].final * vec4(vPos, 1.0f)) * vWeight[3];

	vec3 norm = vec3(0.0f);	
	norm += (mat3(jointTransforms[vWIndex[0]].finalRot) * vNormal) * vWeight[0];
	norm += (mat3(jointTransforms[vWIndex[1]].finalRot) * vNormal) * vWeight[1];
	norm += (mat3(jointTransforms[vWIndex[2]].finalRot) * vNormal) * vWeight[2];
	norm += (mat3(jointTransforms[vWIndex[3]].finalRot) * vNormal) * vWeight[3];
	
	pos = modelTransforms[vTransIndex].model * pos;
	norm = mat3(modelTransforms[vTransIndex].modelRotation) * norm;

	fUV = vUV;
	fNormal = norm;
	fPos = pos.xyz / pos.w;
	gl_Position = orthoFromAABB() * vec4(vPos.xyz, 1.0f);
}