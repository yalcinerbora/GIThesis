#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMap.frag
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

layout (depth_unchanged) out float gl_FragDepth;
layout(early_fragment_tests) in;

void main(void)
{
	// Do Literally nothing depth write is implicit
	gl_FragDepth = gl_FragCoord.z;
}
