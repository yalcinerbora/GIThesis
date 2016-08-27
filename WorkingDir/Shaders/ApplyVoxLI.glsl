#version 430
/*	
	**Apply Voxel Light Buffer Compute Shader**
	
	File Name	: ApplyVoxLI.glsl
	Author		: Bora Yalciner
	Description	:

		Application of the traced light intesity buffer to 
		Deferred Light intensity buffer
*/

#define I_LIGHT_INENSITY layout(rgba16f, binding = 2) restrict

#define T_COLOR layout(binding = 0)

#define U_ON_OFF_SWITCH layout(location = 3)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Uniforms
U_ON_OFF_SWITCH uniform uvec2 onOff;

// Textures
uniform I_LIGHT_INENSITY image2D liTex;

uniform T_COLOR sampler2D giLightIntensity;

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	vec2 uv = (vec2(globalId) + 0.5f) / vec2(imageSize(liTex));

	if(any(greaterThan(ivec2(globalId), imageSize(liTex)))) return;

	vec4 liValue = imageLoad(liTex, ivec2(globalId));
	vec4 giValue = texture(giLightIntensity, uv);

	if(onOff.x == 1) // AO on
	{
		liValue.xyz *= giValue.w;
	}
	if(onOff.y == 1) // GI on
	{
		liValue.xyz += giValue.xyz;
	}
	imageStore(liTex, ivec2(globalId), liValue);
}
