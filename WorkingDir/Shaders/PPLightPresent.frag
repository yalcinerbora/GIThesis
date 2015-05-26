#version 430
/*	
	**Light Present Shader**
	
	File Name	: PPLightPresent.frag 
	Author		: Bora Yalciner
	Description	:

		Combines Intensity Values of lgihts with the actual albedo of surface
*/

// Definitions
#define IN_UV layout(location = 0)
#define OUT_COLOR layout(location = 0)

#define T_COLOR layout(binding = 0)
#define T_INTENSITY layout(binding = 1)

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;

// Textures
uniform T_COLOR sampler2D gBuffColor;
uniform T_INTENSITY sampler2D intensityTex;

void main(void)
{
	fboColor = vec4(texture(gBuffColor, fUV).rgb * 
				    texture(intensityTex, fUV).rgb, 1.0f);
}