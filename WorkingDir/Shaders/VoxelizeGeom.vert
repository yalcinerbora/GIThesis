#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

#define IN_POS layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_UV layout(location = 2)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)
#define OUT_POS layout(location = 2)

#define LU_AABB layout(std430, binding = 3)
#define U_OBJ_ID layout(location = 4)

// Input
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_POS vec3 vPos;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_NORMAL vec3 fNormal;
out OUT_POS vec3 fPos;

// Textures

// Uniforms
U_OBJ_ID uniform uint objId;

LU_AABB buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};

mat4 orthoFromAABB()
{
	// Near Far Left Right Top Bottom
	vec2 nf = vec2(objectAABBInfo[objId].aabbMin.z, objectAABBInfo[objId].aabbMax.z);
	vec2 lr = vec2(objectAABBInfo[objId].aabbMin.x, objectAABBInfo[objId].aabbMax.x);
	vec2 tb = vec2(objectAABBInfo[objId].aabbMax.y, objectAABBInfo[objId].aabbMin.y);

	float xt = - ((lr.y + lr.x) / (lr.y - lr.x));
	float yt = - ((tb.x + tb.y) / (tb.x - tb.y));
	float zt = ((nf.y + nf.x) / (nf.y - nf.x));

	return mat4(2.0f / (lr.y - lr.x),		0.0f,					0.0f,						0.0f,
				0.0f,						2.0f / (tb.x - tb.y),	0.0f,						0.0f,
				0.0f,						0.0f,					-2.0f / (nf.y - nf.x),		0.0f,
				xt,							yt,						zt,							1.0f);
}


void main(void)
{
	fUV = vUV;
	fNormal = vNormal;
	fPos = vPos;
	gl_Position = orthoFromAABB() * vec4(vPos.xyz, 1.0f);
}