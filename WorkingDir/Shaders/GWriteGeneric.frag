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
#define IN_MAT layout(location = 1)
#define IN_TBN layout(location = 2)

// Input
in IN_UV vec2 fUV;
flat in	IN_MAT uint fMatID;
in IN_TBN mat3 fTBN;

// Output
out OUT_RT0 vec4 albedoRGB_metalspecularA;//albedoRG_metalspecularBA;
out OUT_RT1 vec4 normalXYZ_emptyW;

// Textures

// Uniforms

// Here User Defined Material Will Come
void GBufferPopulate(out vec3 fNormal, out vec3 fColor, out vec2 metalSpecular);

void main(void)
{
	// Call Used Define Function
	vec3 fNormal;
	vec3 fColor;
	vec2 metalSpec;

	// User Defines this function (Custom Materials)
	GBufferPopulate(fNormal, fColor, metalSpec);

	// Now do the conversion
	// Albedo YCbCr 4:2:2
	// Stub ....
	albedoRGB_metalspecularA.rgb = fColor.rgb;

	// Normal
	// Stub ....
	normalXYZ_emptyW.rgb = fNormal.xyz;

	// Specular power
	//albedoRG_metalspecularBA.ba = metalSpec;

	// Depth Write is auto, so all done!!!
}

// We will append users shader 
// Users Shader needs to implement this
//void GBufferPopulate(out vec3 fNormal, out vec3 fColor, out vec2 metalSpecular);

/////////////////////////////////////////////
// CUSTOM MATERIAL SHOULD BE APPENDED HERE //
/////////////////////////////////////////////