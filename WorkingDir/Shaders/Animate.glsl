#version 430
/*	
	**Animate Shader**
	
	File Name	: Animate.glsl
	Author		: Bora Yalciner
	Description	:

	Bakes the animation for this current frame on the GPU

*/

#define LU_INV_BIND_POSE layout(std430, binding = 1) readonly
#define LU_BIND_POSE layout(std430, binding = 2) readonly
#define LU_JOINT_HIERARCHY layout(std430, binding = 3) readonly
#define LU_INTERP_JOINTS layout(std430, binding = 4) readonly
#define LU_FINAL_TRANSFORM layout(std430, binding = 5) writeonly

#define BLOCK_SIZE 256

// Uniforms
LU_INV_BIND_POSE buffer InvBindPose
{
	mat4 invBindPose[];
};

LU_BIND_POSE buffer BindPose
{
	vec3 bindPose[3][];
};

LU_JOINT_HIERARCHY buffer Hierarchy
{
	uint parent[];
};

LU_INTERP_JOINTS buffer InterpJoints
{
	vec4 jointData[];
};

LU_FINAL_TRANSFORM buffer FinalTransform
{
	mat4 finalTransform[];
};

mat4 QuatToMat(in vec4 quaternion)
{
	return mat4(1.0f);
}

mat4 TransformGen(in uint poseIndex)
{
	// Quaternion Rotation
	mat4 rot = QuatToMat(jointData[poseIndex] + 1);

	// Scale
	vec3 s = bindPose[2][poseIndex];
	mat4 scale = mat4
	(
		s.x,	0.0f,	0.0f,	0.0f,
		0.0f,	s.y,	0.0f,	0.0f,
		0.0f,	0.0f,	s.z,	0.0f,
		0.0f,	0.0f,	0.0f,	1.0f
	);

	// Translate
	vec3 t = bindPose[0][poseIndex];
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

}
