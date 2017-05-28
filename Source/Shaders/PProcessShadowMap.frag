#version 430
/*	
	**Post Process Shadow Map Shader**
	
	File Name	: PProcessShadowMap.frag 
	Author		: Bora Yalciner
	Description	:

		Passthrough Shader for Shadow Maps
*/

// Definitions
#define IN_UV layout(location = 0)
#define OUT_COLOR layout(location = 0)

#define T_COLOR layout(binding = 0)

#define U_NEAR_FAR layout(location = 1)
#define U_LIGHT_ID layout(location = 4)

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;

// Textures
uniform T_COLOR sampler2DArray shadowMaps;

// Uniforms
U_NEAR_FAR uniform vec2 nearFar;
U_LIGHT_ID uniform uint lightID;

float LinearizeDepth(float depth) 
{   
	return (2.0f * nearFar.x) / (nearFar.y + nearFar.x - depth * (nearFar.y - nearFar.x));
}

void main(void)
{
	float depth = texture(shadowMaps, vec3(fUV, lightID)).x;

	// Do not linearize on Directional Lights
	if(nearFar.x == 0.0f && nearFar.y == 0.0f) fboColor = vec4(depth);
	else fboColor = vec4(LinearizeDepth(depth));
}