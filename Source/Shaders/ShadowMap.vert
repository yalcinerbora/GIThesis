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
#define IN_TRANS_INDEX layout(location = 3)

#define LU_MTRANSFORM layout(std430, binding = 4)

struct ModelTransform
{
	mat4 model;
	mat4 modelRotation;
};

// Input
in IN_POS vec3 vPos;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
invariant gl_Position;

// Uniforms
LU_MTRANSFORM buffer ModelTransformBuffer
{
	ModelTransform modelTransforms[];
};

void main(void)
{
	gl_Position = modelTransforms[vTransIndex].model * vec4(vPos.xyz, 1.0f);
}