#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMapD.geom
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

// Definitions
#define U_FTRANSFORM layout(std140, binding = 0)

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
	mat4 viewRotation;
};

void main(void)
{
	// Dir Light needs only one face we'll write it to +X face
	// Others will be wasted
	gl_Layer = 0;

	// For Each Vertex
	for(unsigned int j = 0; j < gl_in.length(); j++)
	{	
		gl_Position = projection * view * gl_in[j].gl_Position;
		EmitVertex();
	}
	EndPrimitive();
}