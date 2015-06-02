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
#define U_FTRANSFORM layout(std140, binding = 0)
#define U_SHADOW_VIEW layout(std140, binding = 2)

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
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

// Uniforms
U_SHADOW_VIEW uniform ShadowViewMatrices
{
	mat4 viewMatrices[6];
};

void main(void)
{
	// For each layer
	for(unsigned int i = 0; i < 6; i++)
	{
		// Layer 2 is skipped (Area Light Does not Illuminate +Y direction
		if(i != 2)
		{
			// For Each Vertex
			gl_Layer = int(i);
			for(unsigned int j = 0; j < gl_in.length(); j++)
			{	
				if(i == 3)
				{
					// Proj Matrix FOV here should be 90 degrees
					gl_Position = projection * viewMatrices[i] * gl_in[j].gl_Position;
				}
				else
				{
					// Proj Matrix FOV here should be 45 degrees
					// Here view variable holds 45 degree projection matrix
					gl_Position = view * viewMatrices[i] * gl_in[j].gl_Position;
				}
				EmitVertex();
			}
			EndPrimitive();
		}
	}
}