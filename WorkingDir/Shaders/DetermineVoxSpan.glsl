#version 430
#extension GL_ARB_compute_variable_group_size : enable
				
// Definitions
#define LU_OBJECT layout(std430, binding = 3)
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2)

#define U_TOTAL_OBJ_COUNT layout(location = 3)

#define MAX_GRID_DIM 256.0f
#define INCREMENT_FACTOR 0.5f

U_TOTAL_OBJ_COUNT uniform uint objCount;

LU_OBJECT_GRID_INFO buffer GridInfo
{
	struct
	{
		float span;
		uint voxCount;
	} objectGridInfo[];
};

LU_OBJECT buffer AABB
{
	struct
	{
		vec4 aabbMin;
		vec4 aabbMax;
	} objectAABBInfo[];
};
		
layout (local_size_variable) in;
void main(void)
{
	uint globalId = gl_LocalInvocationID.x;
	if(globalId >= objCount) return;

	vec3 dim = objectAABBInfo[globalId].aabbMax.xyz - 
				objectAABBInfo[globalId].aabbMin.xyz;

	float span;
	for(span = 0.5f; 
		dim.x / span > MAX_GRID_DIM ||
		dim.y / span > MAX_GRID_DIM ||
		dim.z / span > MAX_GRID_DIM;
		span += INCREMENT_FACTOR);
	
	objectGridInfo[globalId].span = span;
	objectGridInfo[globalId].voxCount = 0;
}