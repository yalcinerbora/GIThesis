#version 430
/*	
	**BindPoseInverse Shader**
	
	File Name	: GenInvBindPose.glsl
	Author		: Bora Yalciner
	Description	:

	Generates Inverse Bind Pose for each joint

*/

#define LU_INV_BIND_POSE layout(std430, binding = 1) writeonly
#define LU_BIND_POSE layout(std430, binding = 2) readonly
#define LU_JOINT_HIERARCHY layout(std430, binding = 3) readonly

#define BLOCK_SIZE 256

// Uniforms
LU_INV_BIND_POSE buffer InvBindPose
{
	mat4 invBindPose[];
};

LU_BIND_POSE buffer BindPose
{
	vec3 bindPose[][3];
};

LU_JOINT_HIERARCHY buffer Hierarchy
{
	uint parent[];
};

mat4 TransformGen(in uint poseIndex)
{
	// Euler XYZ Rotation
	vec3 r = bindPose[poseIndex][1];
	vec3 c = vec3(cos(r.x), cos(r.y), cos(r.z));
	vec3 s = vec3(sin(r.x), sin(r.y), sin(r.z));
	mat4 rot = mat4
	(
		c.y * c.z,						-c.y * s.z,							s.y,			0.0f,
		c.x * s.z + c.z * s.x * s.y,	c.x * c.z + s.x * s.y * s.z,		-c.y * s.x,		0.0f,
		s.x * s.z + c.x * c.z * s.y,	c.z * s.x + c.x * s.y * s.z,		c.x * c.y,		0.0f,
		0.0f,							0.0f,								0.0f,			1.0f
	);
	//rot = transpose(rot);

	// Scale
	s = bindPose[poseIndex][2];
	mat4 scale = mat4
	(
		s.x,	0.0f,	0.0f,	0.0f,
		0.0f,	s.y,	0.0f,	0.0f,
		0.0f,	0.0f,	s.z,	0.0f,
		0.0f,	0.0f,	0.0f,	1.0f
	);

	// Translate
	vec3 t = bindPose[poseIndex][0];
	mat4 translate = mat4
	(
		1.0f,	0.0f,	0.0f,	0.0f,
		0.0f,	1.0f,	0.0f,	0.0f,
		0.0f,	0.0f,	1.0f,	0.0f,
		t.x,	t.y,	t.z,	1.0f
	);
	return translate * scale * rot;
}

layout (local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uint globalId = gl_GlobalInvocationID.x;

	//
	mat4 invBP = mat4(1.0f);	// mat4(1.0f)
	for(uint i = parent[globalId]; i != 0xFFFFFFFF; i = parent[i])
	{
		// Generate Local Transformation Matrix
		invBP = invBP * inverse(TransformGen(i));
	}
	invBindPose[globalId] = invBP;
}
