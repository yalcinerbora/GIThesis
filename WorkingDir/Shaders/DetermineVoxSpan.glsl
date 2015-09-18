#version 430
				
// Definitions
#define LU_AABB layout(std430, binding = 3) readonly
#define LU_OBJECT_GRID_INFO layout(std430, binding = 2) restrict 

#define U_TOTAL_OBJ_COUNT layout(location = 4)
#define U_MIN_SPAN layout(location = 5)

#define MAX_GRID_DIM 256.0f
#define MAX_GRID_DEPTH 8
//#define MIN_SPAN 1.5f

U_TOTAL_OBJ_COUNT uniform uint objCount;
U_MIN_SPAN uniform float minSpan;

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
	//span = max(span, minSpan);
	float resultSpan = span;
	for(unsigned int i = 1; i <= 0x00000200; i = i << 1)
	{
		if(span <= minSpan * i &&
			span > minSpan * (i >> 1)) 
		{
			resultSpan = i * minSpan; 
			break;
		}
	}

	objectGridInfo[globalId].span = resultSpan;
	objectGridInfo[globalId].voxCount = 0;
}