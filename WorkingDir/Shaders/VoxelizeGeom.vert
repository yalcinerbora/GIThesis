#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.vert
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
*/


#define IN_POS layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_UV layout(location = 2)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)
#define OUT_POS layout(location = 1)

#define U_RENDER layout(binding = 1)

// Input
in IN_POS vec3 vPos;
in IN_NORMAL vec3 vNormal;
in IN_UV vec2 vUV;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
out OUT_NORMAL vec3 fNormal;

// Textures

// Uniforms
U_RENDER uniform ModelTransform
{
	mat4 model;
	mat3 modelRotation;
};

void main(void)
{
	fUV = vUV;
	fNormal = modelRotation * vNormal;
	fPosition =

	gl_Position =  model * vec4(vPos.xyz, 1.0f);
}