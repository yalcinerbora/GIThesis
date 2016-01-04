#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeomCount.glsl
	Author		: Bora Yalciner
	Description	:

		Determines Count of the vox geom
*/

// Definitions
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict

#define U_TOTAL_VOX_DIM layout(location = 3)
#define U_OBJ_ID layout(location = 4)

#define I_VOX_READ layout(rg32ui, binding = 2) restrict readonly

// Input

// Output

// Textures
uniform I_VOX_READ uimage3D voxelData;

// Uniforms
U_OBJ_ID uniform uint objId;
U_TOTAL_VOX_DIM uniform uvec3 voxDim;

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

// Shared Memory
shared uint sLocalVoxCount;

layout (local_size_x = 8, local_size_y = 8, local_size_z = 8) in;
void main(void)
{
	// Init smem
	if(all(equal(gl_LocalInvocationID, uvec3(0)))) sLocalVoxCount = 0;
	//memoryBarrierShared();
	barrier();

	uvec3 voxId = gl_GlobalInvocationID.xyz;
	if(voxId.x >= (voxDim.x) || 
		voxId.y >= (voxDim.y) ||
		voxId.z >= (voxDim.z)) 
		return;

	// Image Fetch
	uvec4 voxData = imageLoad(voxelData, ivec3(voxId));

	// Empty Normal Means its vox is empty
	if(voxData.x != 0xFFFFFFFF ||
		voxData.y != 0xFFFFFFFF)
	{
		atomicAdd(sLocalVoxCount, 1);
	}
	//memoryBarrierShared();
	barrier();

	// Leader of the block will write to global Atomic
	if(all(equal(gl_LocalInvocationID, uvec3(0))))
	{
		atomicAdd(objectGridInfo[objId].voxCount, sLocalVoxCount);
	}
}