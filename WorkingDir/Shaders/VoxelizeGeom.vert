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
	// AABB can be axis aligned plane
	// any of the min max components can be equal (this makes scales zero and we dont want that

	// Near Far Left Right Top Bottom
	// Increase AABB slightly here so that we gurantee entire object will not get culled
	vec3 aabbExpandedMin, aabbExpandedMax;
	aabbExpandedMin = objectAABBInfo[objId].aabbMin.xyz - (abs(0.0000001f * objectAABBInfo[objId].aabbMin.xyz));
	aabbExpandedMax = objectAABBInfo[objId].aabbMax.xyz + (abs(0.0000001f * objectAABBInfo[objId].aabbMax.xyz));

	vec2 nf = vec2(aabbExpandedMin.z, aabbExpandedMax.z);
	vec2 lr = vec2(aabbExpandedMin.x, aabbExpandedMax.x);
	vec2 tb = vec2(aabbExpandedMax.y, aabbExpandedMin.y);

	vec3 diff = vec3((nf.y - nf.x) < 0.00001f ? 0.00001f : (nf.y - nf.x),
					(lr.y - lr.x) < 0.00001f ? 0.00001f : (lr.y - lr.x),
					(tb.x - tb.y) < 0.00001f ? 0.00001f : (tb.x - tb.y)); 

	vec3 translate = vec3(-(lr.y + lr.x),
						  -(tb.x + tb.y),
						  nf.y + nf.x);

	translate = translate / diff.yzx;

	return mat4(2.0f / diff.y,		0.0f,				0.0f,				0.0f,
				0.0f,				2.0f / diff.z,		0.0f,				0.0f,
				0.0f,				0.0f,				-2.0f / diff.x,		0.0f,
				translate.x,		translate.y,		translate.z,		1.0f);
}

void main(void)
{
	fUV = vUV;
	fNormal = vNormal;
	fPos = vPos;
	gl_Position = orthoFromAABB() * vec4(vPos.xyz, 1.0f);
}