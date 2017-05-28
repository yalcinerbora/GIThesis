#version 430
/*	
	**Generic Post Process Shader**
	
	File Name	: PProcessGeneric.frag 
	Author		: Bora Yalciner
	Description	:

		Post Process Pass Shader
		Generic Vertex Shader for post process operations
*/

// Definitions
#define IN_POS layout(location = 0)
#define OUT_UV layout(location = 0)

// Input
in IN_POS vec2 vPos;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
out OUT_UV vec2 fUV;

void main(void)
{
	//		Pos					UV
	//	-1.0f, 3.0f,	-->	0.0f, 2.0f,
	//	3.0f, -1.0f,	-->	2.0f, 0.0f,
	//	-1.0f, -1.0f,	-->	0.0f, 0.0f
	fUV = (vPos + 1.0f) * 0.5f;
	gl_Position = vec4(vPos.xy, 0.0f, 1.0f);
}