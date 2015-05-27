#version 430
/*	
	**Shadow Map Create Shader**
	
	File Name	: ShadowMap.vert
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

// Includes

// Definitions
#define IN_POS layout(location = 0)
#define U_MTRANSFORM layout(std140, binding = 1)

// Input
in IN_POS vec3 vPos;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory

// Uniforms
U_MTRANSFORM uniform ModelTransform
{
	mat4 model;
	mat3 modelRotation;
};

void main(void)
{
	gl_Position = model * vec4(vPos.xyz, 1.0f);
}