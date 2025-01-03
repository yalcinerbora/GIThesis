#version 430
//#extension GL_ARB_shader_draw_parameters : require
/*	
	**G-Buffer Write Shader**
	
	File Name	: GWriteGeneric.vert 
	Author		: Bora Yalciner
	Description	:

		Basic G-buffer write shader...
		Basis of Defferred Shading (Not Optimized)
		Gbuffer is unoptimized
*/

// Includes

// Definitions
#define IN_POS layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_UV layout(location = 2)
#define IN_TRANS_INDEX layout(location = 3)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)

#define U_FTRANSFORM layout(std140, binding = 0)
#define LU_MTRANSFORM layout(std430, binding = 4)

struct ModelTransform
{
	mat4 model;
	mat4 modelRotation;
};

// Input
in IN_POS vec3 vPos;
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_NORMAL vec3 fNormal;
invariant gl_Position;

// Textures

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

LU_MTRANSFORM buffer ModelTransformBuffer
{
	ModelTransform modelTransforms[];
};

void main(void)
{
	fUV = vUV;
	fNormal = mat3(modelTransforms[vTransIndex].modelRotation) * vNormal;

	// Rasterizer
	gl_Position = projection * view * modelTransforms[vTransIndex].model * vec4(vPos.xyz, 1.0f);
}