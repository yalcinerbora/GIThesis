#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/

// Definitions
#define OUT_RT0 layout(location = 0)
#define OUT_RT1 layout(location = 1)
#define OUT_RT2 layout(location = 2)

#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 1)

#define LU_VOXEL layout(std430, binding = 0)
#define LU_VOXEL_RENDER layout(std430, binding = 1)
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_OBJECT layout(std140, binding = 0)

#define T_COLOR layout(binding = 0)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;

// Output

// Textures
uniform T_COLOR sampler2D colorTex;

// Textures

// Uniforms
U_OBJECT uniform Object
{
	vec3 aabbMin;
	vec3 aabbMax;
};

LU_OBJECT_GRID_INFO buffer GridInfo
{
	float span;
	uint voxCount;
};

LU_VOXEL buffer VoxelArray
{
	uvec2 voxelPacked[];
};

LU_VOXEL_RENDER buffer VoxelArrayRender
{
	struct
	{
		vec3 normal;
		uint color;
	} voxelArrayRender[];
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

uvec2 PackVoxelData(in uvec3 voxCoord)
{
	uvec2 vec;
	vec.x = voxCoord.x;
	vec.x |= voxCoord.y << 16;
	vec.y = voxCoord.z;
	vec.y |= 0 << 16;
	return vec;
}

void main(void)
{
	// Get a unique location 
	uint location = atomicAdd(voxCount, 1);

	// Data Packing forming
	vec3 color = texture2D(colorTex, fUV).rgb;
	uint colorPacked = PackColor(color);

	// xy is straightforward
	// z is stored as 0-1 value (unless you change it from api)
	// this form is optimized form generic form is different
	// ogl has its pixel positions defined in midpoint we also compansate that
	float zWindow = (gl_FragCoord.z) * (aabbMax.z - aabbMin.z) / span;
	uvec3 voxelCoord = uvec3(uvec2(gl_FragCoord.xy - vec2(0.5f)), uint(zWindow));

	// Writeback
	voxelPacked[location].xy = PackVoxelData(voxelCoord);
	voxelArrayRender[location].color = colorPacked;
	voxelArrayRender[location].normal = fNormal;
}