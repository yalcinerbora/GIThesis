#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMapP.geom
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

layout(triangles) in;
layout(triangle_strip, max_vertices = 18) out;

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
	for(uint i = 0; i < 6; i++)
	{		
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