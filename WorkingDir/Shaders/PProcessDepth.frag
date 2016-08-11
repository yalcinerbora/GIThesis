#version 430
/*	
	**Post Process Generic Shader**
	
	File Name	: PProcessGeneric.frag 
	Author		: Bora Yalciner
	Description	:

		Pass Trough Shader
*/

// Definitions
#define IN_UV layout(location = 0)
#define OUT_COLOR layout(location = 0)

#define T_COLOR layout(binding = 0)

#define U_NEAR_FAR layout(location = 1)

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;

// Textures
uniform T_COLOR sampler2D gBuffDepth;

// Uniforms
U_NEAR_FAR uniform vec2 nearFar;

float LinearizeDepth(float depth) 
{   
	return (2.0f * nearFar.x) / (nearFar.y + nearFar.x - depth * (nearFar.y - nearFar.x));
}

void main(void)
{
	float depth = texture(gBuffDepth, fUV).x;
	fboColor = vec4(LinearizeDepth(depth));
}