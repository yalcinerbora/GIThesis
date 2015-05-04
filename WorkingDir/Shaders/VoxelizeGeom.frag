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

#define LU_VOXEL layout(location = 0)
#define LU_VOXEL layout(location = 1)
#define LU_OBJECT layout(location = 2)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
in IN_POS vec3 fPos;

// Output

// Textures

// Uniforms
buffer LU_OBJECT Object
{
	vec3 aabbMin;
	vec3 aabbMax;
	uint dataIndex;
}

buffer LU_VOXEL VoxelArray
{
	uvec2 voxelPacked[];
};

buffer LU_VOXEL VoxelArrayRender
{
	struct
	{
		vec3 normal;
		uint color;
	} voxelArrayRender[];
};


const vec4 expand = vec3(255.0f);
const vec4 bitMsk = vec4(0.0f , vec3(1.0f / 256.0));
const vec4 bitShifts = vec4(1.0f) / bitSh;

uint PackColor(vec3 color) 
{
	uint result;
	color *= expand;
    result = uint(color.x) << 0;
	result = uint(color.x) << 8;
	result = uint(color.x) << 16;
    
    return result;
}

void main(void)
{
	vec3 color = texture2D(colorTex, fUV).rgb;
	uint colorPacked = PackColor(color);

	uint location = atomicAdd(dataIndex, 1);

	voxelPacked

}