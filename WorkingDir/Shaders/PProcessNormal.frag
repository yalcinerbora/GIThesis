#version 430
/*	
	**Post Process Generic Shader**
	
	File Name	: PProcessGeneric.frag 
	Author		: Bora Yalciner
	Description	:

		Pass Trough Shader
*/

// Definitions
#define IN_UV layout(location = 0)
#define OUT_COLOR layout(location = 0)

#define T_COLOR layout(binding = 0)

// Input
in IN_UV vec2 fUV;

// Output
out OUT_COLOR vec4 fboColor;

// Textures
uniform T_COLOR usampler2D gBuffNormal;

vec3 UnpackNormal(in uvec2 norm)
{
	vec3 result;
	result.x = ((float(norm.x) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float(norm.y & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm.y << 16));
	return result;
}

void main(void)
{
	vec3 worldNormal = UnpackNormal(texture(gBuffNormal, fUV).xy);
	worldNormal = (worldNormal + 1.0f) * 0.5f;
	fboColor = vec4(worldNormal, 1.0f);
}