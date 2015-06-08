#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMapD.geom
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

#define NUM_SHADOW_CASCADES 4
#define NUM_SHADOW_CASCADE_TRI 12

layout(triangles) in;
layout(triangle_strip, max_vertices = NUM_SHADOW_CASCADE_TRI) out;

// Definitions
#define LU_LIGHT_MATRIX layout(std430, binding = 0)
#define U_LIGHT_ID layout(location = 4)

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
LU_LIGHT_MATRIX buffer LightProjections
{
	struct
	{
		mat4 VPMatrices[6];
	}lightMatrices[];
};


void main(void)
{
	// For each layer
	for(unsigned int i = 0; i < NUM_SHADOW_CASCADES; i++)
	{
		// Dir Light needs only one face
		// Each face will be a cascade
		gl_Layer = int(i);

		// For Each Vertex
		for(unsigned int j = 0; j < gl_in.length(); j++)
		{	
			gl_Position = lightMatrices[lightID].VPMatrices[i] * gl_in[j].gl_Position;
			EmitVertex();
		}
		EndPrimitive();
	}
}