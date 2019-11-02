#version 430
/*	
	**Fast Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry into direct world space
		used to compare transformation between raster based
*/

#define IN_POS layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_UV layout(location = 2)
#define IN_TRANS_INDEX layout(location = 3)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)
#define OUT_POS layout(location = 2)

#define U_VOLUME_SIZE layout(location = 2)
#define U_VOLUME_CORNER layout(location = 3)

#define LU_MTRANSFORM layout(std430, binding = 4)

struct ModelTransform
{
	mat4 model;
	mat4 modelRotation;
};

// Input
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_POS vec3 vPos;
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
	vec4 worldVertex = modelTransforms[vTransIndex].model * vec4(vPos.xyz, 1.0f);
	
	fUV = vUV;
	fNormal = mat3(modelTransforms[vTransIndex].modelRotation) * vNormal;
	fPos = worldVertex.xyz / worldVertex.w;
	
	// Rasterizer
	gl_Position = orthoFromAABB() * worldVertex;
}