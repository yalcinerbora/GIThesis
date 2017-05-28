#version 430
/*	
	**G-Buffer Material Shader**
	
	File Name	: AGenericMaterial.frag 
	Author		: Bora Yalciner
	Description	:

		Simple Color Map Material
		Has to be appended and compiled at the end of the "GWriteGeneric.frag" shader
		in order to be used
*/

// Definitions
#define U_MATERIAL0 layout(binding = 0)
#define U_MATERIAL1 layout(binding = 1)

#define T_COLOR layout(binding = 0)

// Textures
uniform T_COLOR sampler2DArray colorTexArray;

// Uniforms

// Entry
void GBufferPopulate(out vec3 gNormal, out vec3 gColor, out vec2 gMetalSpecular)
{
	vec3 fcolor = texture(colorTexArray, fUV).rgb;
	gNormal = fNormal;
	gMetalSpecular = vec2(0.0f, 0.0f);
}