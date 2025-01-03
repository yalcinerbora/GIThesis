#version 430
/*	
	**Depth Skeletal Pre-Pass Shader**
	
	File Name	: DPass.vert
	Author		: Bora Yalciner
	Description	:

		Depth Prepass
*/

// Includes

// Definitions
#define IN_POS layout(location = 0)
#define IN_TRANS_INDEX layout(location = 3)
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define U_FTRANSFORM layout(std140, binding = 0)
#define LU_MTRANSFORM layout(std430, binding = 4)
#define LU_JOINT_TRANS layout(std430, binding = 6)

#define JOINT_PER_VERTEX 4

struct ModelTransform
{
	mat4 model;
	mat4 modelRotation;
};

struct JointTransform
{
	mat4 final;
	mat4 finalRot;
};

// Input
in IN_POS vec3 vPos;
in IN_WEIGHT vec4 vWeight;
in IN_WEIGHT_INDEX uvec4 vWIndex;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
invariant gl_Position;

// Uniforms
LU_MTRANSFORM buffer ModelTransformBuffer
{
	ModelTransform modelTransforms[];
};

LU_JOINT_TRANS buffer JointTransformBuffer
{
	JointTransform jointTransforms[];
};

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

void main(void)
{
	vec4 pos = vec4(0.0f);
	pos += modelTransforms[vTransIndex].model * jointTransforms[vWIndex[0]].final * vec4(vPos, 1.0f) * vWeight[0];
	pos += modelTransforms[vTransIndex].model * jointTransforms[vWIndex[1]].final * vec4(vPos, 1.0f) * vWeight[1];
	pos += modelTransforms[vTransIndex].model * jointTransforms[vWIndex[2]].final * vec4(vPos, 1.0f) * vWeight[2];
	pos += modelTransforms[vTransIndex].model * jointTransforms[vWIndex[3]].final * vec4(vPos, 1.0f) * vWeight[3];

	// Rasterizer
	gl_Position =  projection * view * pos;
	//gl_Position = projection * view * modelTransforms[vTransIndex].model * pos;
	//gl_Position = projection * view * modelTransforms[vTransIndex].model * vec4(vPos, 1);
	
}