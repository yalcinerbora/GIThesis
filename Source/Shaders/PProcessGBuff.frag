#version 430
/*	
	**Post Process GBuffer Shader**
	
	File Name	: PProcessGBuff.frag 
	Author		: Bora Yalciner
	Description	:

		GBuffer Passthrough Shader for defined values
*/

// Definitions
#define IN_UV layout(location = 0)
#define OUT_COLOR layout(location = 0)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)

#define U_NEAR_FAR layout(location = 1)
#define U_RENDER_MODE layout(location = 2)

// Render Types
#define D_ALBEDO 2
#define S_ALBEDO 3
#define NORMAL 4
#define DEPTH 5

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;

// Uniforms
U_NEAR_FAR uniform vec2 nearFar;
U_RENDER_MODE uniform uint renderMode;

// Textures
uniform T_COLOR sampler2D gBuffer;
uniform T_NORMAL usampler2D gBufferNormal;

vec3 UnpackNormal(in uvec2 norm)
{
	vec3 result;
	result.x = ((float(norm.x) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float(norm.y & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm.y << 16));
	return result;
}

float LinearizeDepth(float depth) 
{   
	return (2.0f * nearFar.x) / (nearFar.y + nearFar.x - depth * (nearFar.y - nearFar.x));
}

void main(void)
{
	vec4 texValue = texture(gBuffer, fUV);
	if(renderMode == D_ALBEDO)
	{
		fboColor = texValue.xyzw;
	}
	else if(renderMode == S_ALBEDO)
	{
		fboColor = texValue.wwww;
	}
	else if(renderMode == NORMAL)
	{
		vec3 worldNormal = UnpackNormal(texture(gBufferNormal, fUV).xy);
		worldNormal = (worldNormal + 1.0f) * 0.5f;
		fboColor = vec4(worldNormal, 1.0f);
	}
	else if(renderMode == DEPTH)
	{
		float depth = LinearizeDepth(texValue.x);
		fboColor = vec4(depth);
	}
}