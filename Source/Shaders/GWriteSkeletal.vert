#version 430
#extension GL_ARB_shader_draw_parameters : require
/*	
	**G-Buffer  Skeletal Mesh Write Shader**
	
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
#define IN_NORMAL layout(location = 1)
#define IN_UV layout(location = 2)
#define IN_TRANS_INDEX layout(location = 3)
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)

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
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_WEIGHT vec4 vWeight;
in IN_WEIGHT_INDEX uvec4 vWIndex;
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

LU_JOINT_TRANS buffer JointTransformBuffer
{
	JointTransform jointTransforms[];
};

uniform vec3 DEBUG_COLORS[4] = 
{
	vec3( 1.0f, 0.0f, 0.0f),
	vec3( 0.0f, 1.0f, 0.0f),
	vec3( 0.0f, 0.0f, 1.0f),
	vec3( 1.0f, 1.0f, 0.0f)
};

void main(void)
{
	// Animations
	vec4 pos = vec4(0.0f);
	pos += (jointTransforms[vWIndex[0]].final * vec4(vPos, 1.0f)) * vWeight[0];
	pos += (jointTransforms[vWIndex[1]].final * vec4(vPos, 1.0f)) * vWeight[1];
	pos += (jointTransforms[vWIndex[2]].final * vec4(vPos, 1.0f)) * vWeight[2];
	pos += (jointTransforms[vWIndex[3]].final * vec4(vPos, 1.0f)) * vWeight[3];

	vec3 norm = vec3(0.0f);	
	norm += (mat3(jointTransforms[vWIndex[0]].finalRot) * vNormal) * vWeight[0];
	norm += (mat3(jointTransforms[vWIndex[1]].finalRot) * vNormal) * vWeight[1];
	norm += (mat3(jointTransforms[vWIndex[2]].finalRot) * vNormal) * vWeight[2];
	norm += (mat3(jointTransforms[vWIndex[3]].finalRot) * vNormal) * vWeight[3];
	
	pos = modelTransforms[vTransIndex].model * pos;
	norm = mat3(modelTransforms[vTransIndex].modelRotation) * norm;		   

	// Rasterizer
	fUV = vUV;
	fNormal = norm;
	gl_Position = projection * view * pos;

	//fNormal = mat3(modelTransforms[vTransIndex].modelRotation) * norm;
	//gl_Position = projection * view * modelTransforms[vTransIndex].model * pos;

	//fNormal = mat3(modelTransforms[vTransIndex].modelRotation) * vNormal;
	//gl_Position = projection * view * modelTransforms[vTransIndex].model * vec4(vPos, 1.0f);
}