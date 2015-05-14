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

#define LU_AABB layout(std430, binding = 3) readonly
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict readonly

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
    result = uint(color.r) << 0;
	result |= uint(color.g) << 8;
	result |= uint(color.b) << 16;
    
    return result;
}

void main(void)
{
	// Data Packing forming
	vec3 color = texture2D(colorTex, fUV).rgb;
	uint colorPacked = PackColor(color);

	// DEBUG
	//colorDebug =  vec4(color.rgb, 1.0f);
	
	// interpolated object space pos
	vec3 voxelCoord = fPos - objectAABBInfo[objId].aabbMin.xyz;
	voxelCoord /= objectGridInfo[objId].span;
	voxelCoord = clamp(voxelCoord, vec3(0.0f), vec3(imageSize(voxelData) - 1.0f));
	
	// TODO: Average the voxel results
	// At the moment it is overwrite
	// Should i need barrier here or some sort of snyc?
	// its ok if these writes atomic, but if vec4 write to tex is not
	// atomic there will be mutated voxels which is bad.
	imageStore(voxelData, ivec3(voxelCoord), vec4(fNormal.xyz, uintBitsToFloat(colorPacked))); 
	//imageStore(voxelData, ivec3(voxelCoord), vec4(color.rgb, uintBitsToFloat(colorPacked))); 
	//imageStore(voxelData, ivec3(0), vec4(1.0f)); 
}