#version 430
/*	
	**Copy Compute Shader**
	
	File Name	: GBufferDepthToR32F.vert
	Author		: Bora Yalciner
	Description	:

		Cuda does not support depth texture copy
		we need to copy depth values of the gbuffer to depth
*/

// Definitions
#define I_DEPTH_R32F layout(r32f, binding = 2) restrict
#define T_DEPTH	layout(binding = 0)
#define U_IMAGE_SIZE layout(location = 4)

// Uniforms
U_IMAGE_SIZE uniform uvec2 imgSize;

// Textures
uniform T_DEPTH sampler2D gBuffDepth;
uniform I_DEPTH_R32F image2D depthR32F;

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	uvec2 globalId = gl_GlobalInvocationID.xy;
	if(any(greaterThanEqual(globalId, imgSize))) return;

	vec2 normCoord = vec2(globalId) / vec2(imgSize);
	float value = texture(gBuffDepth, normCoord).r;
	imageStore(depthR32F, ivec2(globalId), vec4(value)); 
}