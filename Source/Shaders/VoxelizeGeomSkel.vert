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
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)
#define OUT_POS layout(location = 2)
#define OUT_WEIGHT layout(location = 4)
#define OUT_WEIGHT_INDEX layout(location = 5)

#define LU_AABB layout(std430, binding = 3)

#define U_SEGMENT_SIZE layout(location = 2)
#define U_SPLIT_CURRENT layout(location = 7)
#define U_OBJ_ID layout(location = 4)
#define U_SPLAT_RATIO layout(location = 5)

struct AABB
{
	vec4 aabbMin;
	vec4 aabbMax;
};

// Input
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_POS vec3 vPos;
in IN_WEIGHT vec4 vWeight;
in IN_WEIGHT_INDEX uvec4 vWIndex;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_NORMAL vec3 fNormal;
out OUT_POS vec3 fPos;
flat out OUT_WEIGHT vec4 fWeight;
flat out OUT_WEIGHT_INDEX uvec4 fWIndex;

// Textures

// Uniforms
U_OBJ_ID uniform uint objId;
U_SEGMENT_SIZE uniform float segmentSize;
U_SPLIT_CURRENT uniform uvec3 currentSplit;
U_SPLAT_RATIO uniform float splatRatio;

LU_AABB buffer AABBBuffer
{
	AABB objectAABBInfo[];
};

mat4 orthoFromAABB()
{
	// AABB can be axis aligned plane
	// any of the min max components can be equal (this makes scales zero and we dont want that)
	
	// Project only the required segment of the object
	vec3 aabbMin = objectAABBInfo[objId].aabbMin.xyz;
	vec3 aabbMax = objectAABBInfo[objId].aabbMax.xyz;
	// Increase AABB slightly here so that we gurantee entire object will not get culled
	aabbMin -= (abs(0.00100001f * aabbMin));
	aabbMax += (abs(0.00100001f * aabbMax));

	// Divide it into 
	aabbMin += vec3(currentSplit) * vec3(segmentSize);
	aabbMax = min(aabbMax - aabbMin, vec3(segmentSize));
	aabbMax += aabbMin;

	// Near Far Left Right Top Bottom
	vec2 lr = vec2(aabbMin.x, aabbMax.x);
	vec2 tb = vec2(aabbMin.y, aabbMax.y);
	vec2 nf = vec2(aabbMax.z, aabbMin.z);

	vec3 diff = vec3(aabbMax.x - aabbMin.x,
					 aabbMax.y - aabbMin.y,
					 aabbMax.z - aabbMin.z);
	diff = max(diff, 0.00001f);

	vec3 translate = vec3(-lr.y - lr.x,
						  -tb.x - tb.y,
						  +nf.y + nf.x);
	translate = translate / diff.xyz;

	return mat4(2.0f / diff.x,		0.0f,				0.0f,				0.0f,
				0.0f,				2.0f / diff.y,		0.0f,				0.0f,
				0.0f,				0.0f,				-2.0f / diff.z,		0.0f,
				translate.x,		translate.y,		translate.z,		1.0f);
}

void main(void)
{
	fUV = vUV;
	fNormal = vNormal;
	fPos = vPos;
	fWeight = vWeight;
	fWIndex = vWIndex;
	gl_Position = orthoFromAABB() * vec4(vPos.xyz, 1.0f);
}