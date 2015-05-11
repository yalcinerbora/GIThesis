#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

// Definitions
#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)

#define LU_AABB layout(std430, binding = 3)
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define T_COLOR layout(binding = 0)
#define I_VOX_WRITE layout(rgba32f, binding = 2) restrict writeonly
#define U_OBJ_ID layout(location = 4)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
in IN_POS vec3 fPos;

// Output
out vec4 colorDebug;

// Textures
uniform T_COLOR sampler2D colorTex;
uniform I_VOX_WRITE image3D voxelData;

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

uint PackColor(vec3 color) 
{
	uint result;
	color *= vec3(255.0f);
    result = uint(color.x) << 0;
	result = uint(color.x) << 8;
	result = uint(color.x) << 16;
    
    return result;
}

void main(void)
{
	// Data Packing forming
	vec3 color = texture2D(colorTex, fUV).rgb;
	uint colorPacked = PackColor(color);

	// DEBUG
	colorDebug =  vec4(color.rgb, 1.0f);

	// xy is straightforward
	// z is stored as 0-1 value (unless you change it from api)
	// this form is optimized form generic form is different
	// ogl has its pixel positions defined in midpoint we also compansate that
	
	vec3 voxelCoord = fPos - objectAABBInfo[objId].aabbMin.xyz;
	voxelCoord /= objectGridInfo[objId].span;

	//uvec3 voxCoords = (aabbSize - fPos)
	//				(objectAABBInfo[objId].aabbMax.z - objectAABBInfo[objId].aabbMin.z) / 
	//				objectGridInfo[objId].span;
	//uvec3 voxelCoord = uvec3(uvec2(gl_FragCoord.xy - vec2(0.5f)), zWindow);//uint(0));
	
	// TODO: Average the voxel results
	// At the moment it is overwrite
	//imageStore(voxelData, ivec3(voxelCoord), vec4(fNormal.xyz, uintBitsToFloat(colorPacked))); 
	imageStore(voxelData, ivec3(voxelCoord), vec4(color.xyz, uintBitsToFloat(colorPacked))); 
}