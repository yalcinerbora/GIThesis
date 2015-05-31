#version 430
/*	
	**Depth Pre-Pass Shader**
	
	File Name	: DPass.vert
	Author		: Bora Yalciner
	Description	:

		Dpeth Prepass
*/

// Includes

// Definitions
#define IN_POS layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_MTRANSFORM layout(std140, binding = 1)

// Input
in IN_POS vec3 vPos;

// Output
out gl_PerVertex {invariant vec4 gl_Position;};	// Mandatory

// Uniforms
U_MTRANSFORM uniform ModelTransform
{
	mat4 model;
	mat3 modelRotation;
};

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
	mat4 viewRotation;
};

void main(void)
{
	gl_Position = projection * view * model * vec4(vPos.xyz, 1.0f);
}