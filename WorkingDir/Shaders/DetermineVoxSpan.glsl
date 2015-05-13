#version 430
				
// Definitions
#define LU_AABB layout(std430, binding = 3)
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_TOTAL_OBJ_COUNT layout(location = 4)

#define MAX_GRID_DIM 128.0f
#define MIN_SPAN 1.0f

U_TOTAL_OBJ_COUNT uniform uint objCount;

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

LU_AABB buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};
		
layout (local_size_x = 128, local_size_y = 1, local_size_z = 1) in;
void main(void)
{
	uint globalId = gl_GlobalInvocationID.x;
	if(globalId >= objCount) return;

	vec3 dim = objectAABBInfo[globalId].aabbMax.xyz - 
				objectAABBInfo[globalId].aabbMin.xyz;

	dim.xyz = dim.xyz / MAX_GRID_DIM;
	float span = max(max(dim.x, dim.y), dim.z);
	span = max(span, MIN_SPAN);
	
	objectGridInfo[globalId].span = span;
	objectGridInfo[globalId].voxCount = 0;
}