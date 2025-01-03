#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMapD.geom
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

#define NUM_SHADOW_CASCADES 6
#define NUM_SHADOW_CASCADE_TRI 18

layout(triangles) in;
layout(triangle_strip, max_vertices = NUM_SHADOW_CASCADE_TRI) out;

// Definitions
#define LU_LIGHT_MATRIX layout(std430, binding = 0)
#define U_LIGHT_ID layout(location = 4)

struct LightProjections
{
	mat4 VPMatrices[6];
};

// Input
in gl_PerVertex 
{
    vec4  gl_Position;
    //float gl_PointSize;
    //float gl_ClipDistance[];
} gl_in[];

// Output
out gl_PerVertex 
{
    vec4  gl_Position;
    //float gl_PointSize;
    //float gl_ClipDistance[];
};

// Unfiorms
U_LIGHT_ID uniform uint lightID;

// SSBO
LU_LIGHT_MATRIX buffer LightProjectionBuffer
{
	LightProjections lightMatrices[];
};


void main(void)
{
	// For each layer
	for(uint i = 0; i < NUM_SHADOW_CASCADES; i++)
	{
		// Dir Light needs only one face
		// Each face will be a cascade
		gl_Layer = int(i);

		// For Each Vertex
		for(uint j = 0; j < gl_in.length(); j++)
		{	
			gl_Position = lightMatrices[lightID].VPMatrices[i] * gl_in[j].gl_Position;
			EmitVertex();
		}
		EndPrimitive();
	}
}