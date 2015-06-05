#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMapA.geom
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

// Uniforms
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
	for(unsigned int i = 0; i < 6; i++)
	{
		// Layer 2 is skipped (Area Light Does not Illuminate +Y direction
		if(i == 2)
			continue;

		// For Each Vertex
		gl_Layer = int(i);
		for(unsigned int j = 0; j < gl_in.length(); j++)
		{	
			if(i == 3)
			{
				// Proj Matrix FOV here should be 90 degrees
				gl_Position = lightMatrices[lightID].VPMatrices[i] * gl_in[j].gl_Position;
			}
			else
			{
				// Proj Matrix FOV here should be 45 degrees
				// Here view variable holds 45 degree projection matrix
				gl_Position = lightMatrices[lightID].VPMatrices[i] * gl_in[j].gl_Position;
			}
			EmitVertex();
		}
		EndPrimitive();
	}
}