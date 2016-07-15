#version 430
/*	
	**Interp Animation Shader**
	
	File Name	: InterpAnimation.glsl
	Author		: Bora Yalciner
	Description	:

	Generates Interpolated Animation Values for each key etc.

*/

#define LU_ANIMATION layout(std430, binding = 0) readonly
#define LU_INTERP_JOINTS layout(std430, binding = 4) writeonly

#define U_TIMES layout(location = 0)
#define U_COUNTS layout(location = 1)

U_TIMES uniform uvec2 frames;	// X toFrame Y fromFrame
U_COUNTS uniform uvec2 counts;	// X Frame Count Y Bone Count

#define BLOCK_SIZE 256

// Uniforms
LU_ANIMATION buffer Animation
{
	vec4 animation[];
};

LU_INTERP_JOINTS buffer InterpJoints
{
	vec4 outJoints[];
};

vec4 QuatMult(in vec4 quat1, in vec4 quat2)
{
	vec4 q1 = quat1.yzwx;
	vec4 q2 = quat2.yzwx;
	return vec4(q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z,		// W
				q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y,		// X
				q1.w * q2.y + q1.y * q2.w + q1.z * q2.x - q1.x * q2.z,		// Y
				q1.w * q2.z + q1.z * q2.w + q1.x * q2.y - q1.y * q2.x);		// Z
}

vec3 Rotate(in vec4 quaternion, in vec3 vector)
{
	vec4 qInv = vec4(quaternion.x, -quaternion.y, -quaternion.z, -quaternion.w);
	vec4 result = QuatMult(vec4(0.0f, vector.x, vector.y, vector.z), qInv);
	return QuatMult(quaternion, result).xyz;
}

layout (local_size_x = BLOCK_SIZE, local_size_y = 1, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uint globalId = gl_GlobalInvocationID.x;
}