#version 430
#extension GL_ARB_shader_draw_parameters : require
/*	
	**Shadow Map  Skeletal Mesh Write Shader**
	
	File Name	: GWriteSkeletal.vert 
	Author		: Bora Yalciner
	Description	:

		Skeletal Mesh G-buffer write shader...
		Basis of Defferred Shading (Not Optimized)
		Gbuffer is unoptimized
*/

// Includes

// Definitions
#define IN_POS layout(location = 0)
#define IN_TRANS_INDEX layout(location = 3)
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define LU_MTRANSFORM layout(std430, binding = 4) readonly
#define LU_JOINT_TRANS layout(std430, binding = 6) readonly

#define JOINT_PER_VERTEX 4

// Input
in IN_POS vec3 vPos;
in IN_WEIGHT vec4 vWeight;
in IN_WEIGHT_INDEX uvec4 vWIndex;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {invariant vec4 gl_Position;};	// Mandatory

// Textures

// Uniforms
LU_MTRANSFORM buffer ModelTransform
{
	struct
	{
		mat4 model;
		mat4 modelRotation;
	} modelTransforms[];
};

LU_JOINT_TRANS buffer JointTransforms
{
	struct
	{
		mat4 final;
		mat4 finalRot;
	} jointTransforms[];
};

void main(void)
{
	// Animations
	vec4 pos = vec4(0.0f);
	pos += (jointTransforms[vWIndex[0]].final * modelTransforms[vTransIndex].model * vec4(vPos, 1.0f)) * vWeight[0];
	pos += (jointTransforms[vWIndex[1]].final * modelTransforms[vTransIndex].model * vec4(vPos, 1.0f)) * vWeight[1];
	pos += (jointTransforms[vWIndex[2]].final * modelTransforms[vTransIndex].model * vec4(vPos, 1.0f)) * vWeight[2];
	pos += (jointTransforms[vWIndex[3]].final * modelTransforms[vTransIndex].model * vec4(vPos, 1.0f)) * vWeight[3];
	
	// Rasterizer
	gl_Position = pos;
}