#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeomCount.glsl
	Author		: Bora Yalciner
	Description	:

		Determines Count of the vox geom
*/

// Definitions
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)
#define U_VOX_SLICE layout(location = 5)

#define I_VOX_READ layout(rgba32f, binding = 2) restrict readonly

// Input

// Output

// Textures
uniform I_VOX_READ image3D voxelData;

// Uniforms
U_OBJ_ID uniform uint objId;
U_TOTAL_VOX_DIM uniform uvec3 voxDim;
U_VOX_SLICE uniform uvec2 voxSlice;

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main(void)
{
	uvec3 voxId;
	uint localBlockID = gl_WorkGroupID.x % voxSlice.y;
	voxId.xy  = gl_LocalInvocationID.xy + 
				uvec2( localBlockID % voxSlice.x, localBlockID / voxSlice.x ) * uvec2(32); 
	voxId.z = gl_WorkGroupID.x / voxSlice.y;

	if(voxId.x >= voxDim.x || 
		voxId.y >= voxDim.y ||
		voxId.z >= voxDim.z) return;

	vec4 voxData = imageLoad(voxelData, ivec3(voxId));

	// Empty Normal Means its vox is empty
	if(voxData.x != 0.0f ||
		voxData.y != 0.0f ||
		voxData.z != 0.0f)
	{
		atomicAdd(objectGridInfo[objId].voxCount, 1);
	}
}