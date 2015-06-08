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
#define IN_TRANS_INDEX layout(location = 3)

#define U_FTRANSFORM layout(std140, binding = 0)
#define LU_MTRANSFORM layout(std430, binding = 4)

// Input
in IN_POS vec3 vPos;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {invariant vec4 gl_Position;};	// Mandatory

// Uniforms
LU_MTRANSFORM buffer ModelTransform
{
	struct
	{
		mat4 model;
		mat4 modelRotation;
	} modelTransforms[];
};

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

void main(void)
{
	gl_Position = projection * view * modelTransforms[vTransIndex].model * vec4(vPos.xyz, 1.0f);
}