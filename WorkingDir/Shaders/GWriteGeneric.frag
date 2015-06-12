#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: GWriteGeneric.frag 
	Author		: Bora Yalciner
	Description	:

		Basic G-buffer write shader...
		Basis of Defferred Shading (Not Optimized)
		Gbuffer is unoptimized
*/

// Definitions
#define OUT_RT0 layout(location = 0)
#define OUT_RT1 layout(location = 1)
#define OUT_RT2 layout(location = 2)

#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)

// Input
in IN_UV vec2 fUV;
in IN_NORMAL vec3 fNormal;
layout(early_fragment_tests) in;

// Output
out OUT_RT0 vec4 albedoRGB_specPowA;
out OUT_RT1 uvec2 normalXY;

// Textures

// Uniforms

// Here User Defined Material Will Come
void GBufferPopulate(out vec3 gNormal, out vec3 gColor, out float gSpec);

uvec2 PackNormal(in vec3 normal)
{
	// 1615 XY Format
	// 32 bit format LS 16 bits are X
	// MSB is the sign of Z
	// Rest is Y
	// both x and y is SNORM types
	uvec2 result = uvec2(0.0f);
	result.x = uint((normal.x * 0.5f + 0.5f) * 0xFFFF);
	result.y = uint((normal.y * 0.5f + 0.5f) * 0x7FFF);
	result.y |= (floatBitsToUint(normal.z) >> 16) & 0x00008000;
	return result;
}

void main(void)
{
	// Call Used Define Function
	vec3 gColor;
	vec3 gNormal;
	float gSpec;

	// User Defines this function (Custom Materials)
	GBufferPopulate(gNormal, gColor, gSpec);

	// GBuffer Write
	albedoRGB_specPowA.rgb = gColor;
	albedoRGB_specPowA.a = gSpec;
	normalXY = PackNormal(normalize(gNormal));
	// Depth Write is auto, so all done!!!
}

// We will append users shader 
// Users Shader needs to implement this
//void GBufferPopulate(out vec3 fNormal, out vec3 fColor, out vec2 metalSpecular);
/////////////////////////////////////////////
// CUSTOM MATERIAL SHOULD BE APPENDED HERE //
/////////////////////////////////////////////
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
#define U_MATERIAL0 layout(binding = 1)
#define U_MATERIAL1 layout(binding = 2)

#define T_COLOR layout(binding = 0)

// Textures
uniform T_COLOR sampler2D colorTex;

// Uniforms
// Entry
void GBufferPopulate(out vec3 gNormal, out vec3 gColor, out float gSpec)
{
	gColor = texture(colorTex, fUV).rgb;
	gNormal = fNormal;
	gSpec = 0.02f;
}