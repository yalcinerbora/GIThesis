#version 430
/*	
	**Depth Prepass Shader**
	
	File Name	: DPass.frag
	Author		: Bora Yalciner
	Description	:

		Depth prepass Shader
*/

layout (depth_unchanged) out float gl_FragDepth;
layout(early_fragment_tests) in;

void main(void)
{
	// Do Literally nothing depth write is implicit
}