#version 430
#extension GL_ARB_shader_draw_parameters : require
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
#define IN_UV layout(location = 1)
#define IN_NORMAL layout(location = 2)
#define IN_TANGENT layout(location = 3)
#define IN_SIGN layout(location = 4)

#define OUT_UV layout(location = 0)
#define OUT_MAT layout(location = 1)
#define OUT_TBN layout(location = 2)

#define LU_RENDER layout(binding = 3)
#define LU_SHRINK_ID layout(binding = 4)

#define U_FTRANSFORM layout(binding = 0)

// Input
in IN_POS vec3 vPos;
in IN_UV vec2 vUV;
in IN_NORMAL vec2 vNormal;
in IN_TANGENT vec2 vTangent;
in IN_SIGN int vTBNSign;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;
flat out OUT_MAT uint fMatID;
out OUT_TBN mat3 fTBN;

// Textures

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
	mat3 viewRotation;
};

LU_RENDER buffer RenderData
{
	struct
	{
		mat4 model;
		mat3 modelRotation;
		int meshID;
		int materialID;
	} perRenderData [];
};

LU_SHRINK_ID buffer ShrinkSortedID
{
	uint shrinkSortedID[];
};

void main(void)
{
	// Send to Fragment Shader
	fUV = vUV;
	fMatID = perRenderData[shrinkSortedID[gl_DrawIDARB + gl_InstanceID]].materialID;

	// Construct tbn matrix	
	int ns = (vTBNSign >> 0) & 0x0001;
	int ts = (vTBNSign >> 1) & 0x0001;
	int bs = (vTBNSign >> 2) & 0x0001;

	vec3 n = vec3(vNormal.xy, sqrt(1.0f - dot(vNormal.xy, vNormal.xy)) * (1 - 2 * ns));
    vec3 t = vec3(vTangent.xy, sqrt(1.0f - dot(vTangent.xy, vTangent.xy)) * (1 - 2 * ts));
    vec3 b = cross(t, n) * (1 - 2 * bs);

	// tbn -> tangent space to object space
	mat3 tbn = mat3(t.x, t.y, t.z,
					b.x, b.y, b.z,
					n.x, n.y, n.z);
	tbn = viewRotation * perRenderData[shrinkSortedID[gl_DrawIDARB + gl_InstanceID]].modelRotation * tbn;
	// fTBN -> tangent space to world space
	fTBN = tbn;

	// Rasterizer
	gl_Position = projection * view * perRenderData[shrinkSortedID[gl_DrawIDARB + gl_InstanceID]].model  * vec4(vPos.xyz, 1.0f);
}