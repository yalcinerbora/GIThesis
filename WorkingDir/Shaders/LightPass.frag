#version 430
/*	
	**Lightpass Shader**
	
	File Name	: LightPass.frag 
	Author		: Bora Yalciner
	Description	:

		LightPass Shader
*/

// Definitions
#define IN_INDEX layout(location = 0)

#define OUT_COLOR layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)

#define LU_LIGHT layout(std430, binding = 1)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)
#define T_SHADOW layout(binding = 3)

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f
#define GI_LIGHT_AREA 2.0f

#define GI_ONE_OVER_2_PI 0.159154

// Input
flat in IN_INDEX int fIndex;

// Output
out OUT_COLOR vec4 fboColor;

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
	mat4 viewRotation;

	vec4 camPos;		// To Calculate Eye
	ivec4 viewport;		// Viewport Params
};

U_INVFTRANSFORM uniform InverseFrameTransform
{
	mat4 invView;
	mat4 invProjection;
	mat4 invViewRotation;
};

LU_LIGHT buffer LightParams
{
	// If Position.w == 0, Its point light
	//		makes direction obselete
	// If Position.w == 1, Its directional light
	//		makes position.xyz obselete
	//		direction.w is obselete
	//		color.a is obselete
	// If Position.w == 2, Its area light
	//
	struct
	{
		vec4 position;			// position.w is the light type
		vec4 direction;			// direction.w holds shadow map index
		vec4 color;				// color.a is effecting radius
	} lightParams[];
};

// Textures
uniform T_COLOR sampler2D gBuffColor;
uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;
uniform T_SHADOW sampler2DArrayShadow shadowMaps;

vec3 DepthToWorld()
{
	// Converts Depthbuffer Value to World Coords
	// First Depthbuffer to Screen Space
	vec3 ssPos;
	// ... 
	// ... 

	// From Screen Space to World Space
	return (invView * invProjection * vec4(ssPos, 1.0f)).xyz;
}

vec3 UnpackNormal(uvec2 norm)
{
	vec3 result;
	result.x = (float(norm.x) / 65536.0f * 2.0f) - 1.0f;
	result.y = (float(norm.y & 0x7FFF) / 32768.0f * 2.0f) - 1.0f;
	result.z = sqrt(result.x * result.x + result.y * result.y);
	result.z *= ((norm.y & 0x8000) >> 15 == 1) ? -1.0f : 1.0f;
	return result;
}

vec3 PhongBDRF(in vec3 worldPos)
{
	vec3 lightIntensity = vec3(0.0f);

	// UV Coords
	vec2 gBuffUV = (gl_FragCoord.xy - vec2(0.5f)) / viewport.xy;
	vec2 shadowUV = vec2(0.0f);

	// Check Light Occulusion to prevent unnecesary calculation
	// (At least on some warps)
	float shadowIntensity = texture(shadowMaps, vec4(shadowUV, float(fIndex) , gl_FragCoord.z));
	if(shadowIntensity == 0.0f)
		return lightIntensity;

	// Phong BDRF Calculation
	// Outputs intensity multiplier for each channel (rgb)
	// Diffuse is Lambert
	// We store normals in world space in GBuffer
	vec3 worldNormal = UnpackNormal(texture(gBuffNormal, gBuffUV).xy);
	vec3 worldEye = camPos.xyz - worldPos;
	
	vec3 worldLight;
	if(lightParams[fIndex].position.w != GI_LIGHT_POINT)
		worldLight = -lightParams[fIndex].direction.xyz;
	else
		worldLight = lightParams[fIndex].position.xyz - worldPos;
	worldLight = normalize(worldLight);
	worldNormal = normalize(worldNormal);
	worldEye = normalize(worldEye);
	vec3 worldReflect = reflect(worldLight, worldNormal);

	// Diffuse Factor
	// Lambert Diffuse Model
	lightIntensity = vec3(dot(worldLight, worldNormal));

	// Specular
	float specPower = texture2D(gBuffColor, gBuffUV).a * 128.0f;
	lightIntensity += ((specPower + 2.0f) * GI_ONE_OVER_2_PI) * 
						vec3(pow(dot(worldEye, worldReflect), specPower));

	// Light Falloff Calculation
	// TODO:

	// Colorize
	lightIntensity *= lightParams[fIndex].color.rgb;
	lightIntensity *= shadowIntensity;
	return lightIntensity;
}

void main(void)
{
	// Do Light Calculation
	vec3 lightIntensity = PhongBDRF(DepthToWorld());
	
	// Additive Blending will take care of the rest
	fboColor = vec4(lightIntensity, 1.0f);
}