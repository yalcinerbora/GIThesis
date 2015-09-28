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
#define I_VOX_WRITE layout(rgba16ui, binding = 2) restrict writeonly
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

uvec2 PackColor(vec3 color) 
{
	uvec2 result;
	color *= vec3(255.0f);
    result.x = uint(color.g) << 8;
	result.x |= uint(color.r);
	result.y = uint(/*color.a*/0) << 16;
	result.y |= uint(color.b) << 0;
    
    return result;
}

uvec2 PackNormal(in vec3 normal)
{
	// 1615 XY Format
	// 32 bit format LS 16 bits are X
	// MSB is the sign of Z
	// Rest is Y
	// both x and y is SNORM types
	uvec2 result = uvec2(0.0f);
	result.x = uint((normal.x * 0.5f + 0.5f) * 0xFFFF);
	result.y = uint((normal.y * 0.5f + 0.5f) * 0x7FFF);
	result.y |= (floatBitsToUint(normal.z) >> 16) & 0x00008000;
	return result;
}

void main(void)
{
	// Data Packing forming
	vec3 color = texture2D(colorTex, fUV).rgb;

	// DEBUG
	//colorDebug =  vec4(color.rgb, 1.0f);
	
	// interpolated object space pos
	vec3 voxelCoord = (fPos - objectAABBInfo[objId].aabbMin.xyz) / objectGridInfo[objId].span;
	ivec3 voxelCoordInt = ivec3(voxelCoord + 0.5f);
	voxelCoordInt = clamp(voxelCoordInt, ivec3(0), imageSize(voxelData) - 1);
	
	if(all(lessThan(voxelCoordInt, imageSize(voxelData))) &&
		all(greaterThanEqual(voxelCoordInt, ivec3(0))))
	{
		// TODO: Average the voxel results
		// At the moment it is overwrite
		// Should i need barrier here or some sort of snyc?
		// its ok if these writes atomic, but if vec4 write to tex is not
		// atomic there will be mutated voxels which is bad.
		imageStore(voxelData, voxelCoordInt, uvec4(PackNormal(fNormal.xyz), PackColor(color))); 
		//imageStore(voxelData, ivec3(voxelCoord), vec4(color.rgb, uintBitsToFloat(colorPacked))); 
		//imageStore(voxelData, ivec3(0), vec4(1.0f)); 
	}
}