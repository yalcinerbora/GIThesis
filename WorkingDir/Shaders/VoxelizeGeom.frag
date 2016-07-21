#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.frag
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)

#define LU_AABB layout(std430, binding = 3) readonly
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict readonly

#define T_COLOR layout(binding = 0)
#define I_VOX_WRITE layout(rg32ui, binding = 2) restrict writeonly

#define U_OBJ_ID layout(location = 4)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
in IN_POS vec3 fPos;

// Output
out vec4 colorDebug;

// Textures
uniform T_COLOR sampler2D colorTex;
uniform I_VOX_WRITE uimage3D voxelData;

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

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

uint PackColor(in vec3 color, in float specularity) 
{
	// 8bit xyzw UNORM
	return packUnorm4x8(vec4(color, specularity));
}

uint PackNormal(in vec3 normal)
{
	// 8bit xyz SNORM
	return packSnorm4x8(vec4(normal, 0.0f));
}

void main(void)
{
	// Data Packing forming
	vec3 color = texture2D(colorTex, fUV).rgb;

	// interpolated object space pos
	vec3 voxelCoord = floor((fPos - objectAABBInfo[objId].aabbMin.xyz) / objectGridInfo[objId].span);

	// TODO: Average the voxel results
	// At the moment it is overwrite
	imageStore(voxelData, 
			   ivec3(voxelCoord), 
			   uvec4(PackNormal(fNormal.xyz), PackColor(color, 1.0f), 0.0f, 0.0f));
}