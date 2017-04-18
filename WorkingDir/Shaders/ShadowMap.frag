#version 430
/*	
	**G-Buffer Write Shader**
	
	File Name	: ShadowMap.frag
	Author		: Bora Yalciner
	Description	:

		Shadowmap Creation Shader
*/

//layout (depth_unchanged) out float gl_FragDepth;
layout(early_fragment_tests) in;

#define OUT_COLOR layout(location = 0)

out OUT_COLOR float depthOut;

void main(void)
{
	depthOut = gl_FragCoord.z;
}
