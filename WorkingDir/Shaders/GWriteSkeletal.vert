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
#define LU_JOINT_TRANS layout(std430, binding = 5)

#define JOINT_PER_VERTEX 4

// Input
in IN_POS vec3 vPos;
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;
in IN_WEIGHT_INDEX uvec4 vWIndex;
in IN_WEIGHT vec4 vWeight;
in IN_TRANS_INDEX uint vTransIndex;

// Output
out gl_PerVertex {invariant vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_NORMAL vec3 fNormal;

// Textures

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

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
	mat4 jointTransforms[];
};

void main(void)
{
	// Animations
	vec4 vertPos = vec4(0.0f);
	vec4 vertNorm = vec4(0.0f);
	for(uint i = 0; i < JOINT_PER_VERTEX; i++)
	{
		vertPos += (jointTransforms[vWIndex[i]] * vec4(vPos, 1.0f)) * vWeight[i];
		vertNorm += (jointTransforms[vWIndex[i]] * vec4(vNormal, 0.0f)) * vWeight[i];
	}
	
	// Rasterizer
	fUV = vUV;
	fNormal = mat3(modelTransforms[vTransIndex].modelRotation) * vertNorm.xyz;
	gl_Position = projection * view * modelTransforms[vTransIndex].model * 
				  vec4(vertPos.xyz, 1.0f);
}