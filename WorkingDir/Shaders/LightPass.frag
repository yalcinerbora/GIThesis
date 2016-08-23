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

#define U_SHADOW_MIP_COUNT layout(location = 0)
#define U_SHADOW_MAP_WH layout(location = 1)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)

#define LU_LIGHT layout(std430, binding = 1)
#define LU_LIGHT_MATRIX layout(std430, binding = 0)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)
#define T_SHADOW layout(binding = 3)
#define T_SHADOW_DIR layout(binding = 4)

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f
#define GI_LIGHT_AREA 2.0f

#define GI_ONE_OVER_PI 0.318309f

// Input
flat in IN_INDEX uint fIndex;
layout(early_fragment_tests) in;

// Output
out OUT_COLOR vec4 fboColor;

// Uniforms
U_SHADOW_MIP_COUNT uniform uint shadowMipCount;
U_SHADOW_MAP_WH uniform uint shadowMapWH;

U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
};

U_INVFTRANSFORM uniform InverseFrameTransform
{
	mat4 invViewProjection;

	vec4 camPos;		// To Calculate Eye
	vec4 camDir;		// To Calculate Eye
	ivec4 viewport;		// Viewport Params
	vec4 depthNearFar;	// depth range params (last two unused)
};

LU_LIGHT_MATRIX buffer LightProjections
{
	struct
	{
		mat4 VPMatrices[6];
	}lightMatrices[];
};

LU_LIGHT buffer LightParams
{
	// If Position.w == 0, Its point light
	//		makes direction obselete
	// If Position.w == 1, Its directional light
	//		makes position.xyz obselete
	//		color.a is obselete
	// If Position.w == 2, Its area light
	//
	struct
	{
		vec4 position;			// position.w is the light type
		vec4 direction;			// direction.w is areaLight w/h ratio
		vec4 color;				// color.a is effecting radius
	} lightParams[];
};

// Textures
uniform T_COLOR sampler2D gBuffColor;
uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;
//uniform T_SHADOW samplerCubeArrayShadow shadowMaps;
//uniform T_SHADOW_DIR sampler2DArrayShadow shadowMapsDir;
uniform T_SHADOW samplerCubeArray shadowMaps;
uniform T_SHADOW_DIR sampler2DArray shadowMapsDir;

vec3 DepthToWorld()
{
	vec2 gBuffUV = vec2(gl_FragCoord.xy - viewport.xy /*- vec2(0.5f)*/) / viewport.zw;

	// Converts Depthbuffer Value to World Coords
	// First Depthbuffer to Screen Space
	vec3 ndc = vec3(gBuffUV, texture(gBuffDepth, gBuffUV).x);
	ndc.xy = 2.0f * ndc.xy - 1.0f;
	ndc.z = ((2.0f * (ndc.z - depthNearFar.x) / (depthNearFar.y - depthNearFar.x)) - 1.0f);

	// Clip Space
	vec4 clip;
	clip.w = projection[3][2] / (ndc.z - (projection[2][2] / projection[2][3]));
	clip.xyz = ndc * clip.w;

	// From Clip Space to World Space
	return (invViewProjection * clip).xyz;
}

vec3 UnpackNormal(in uvec2 norm)
{
	vec3 result;
	result.x = ((float(norm.x) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float(norm.y & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm.y << 16));
	return result;
}

vec4 CalculateShadowUV(in vec3 worldPos)
{
	float viewIndex = 0.0f;
	vec3 lightVec;
	if(lightParams[fIndex].position.w == GI_LIGHT_POINT ||
		lightParams[fIndex].position.w == GI_LIGHT_AREA)
	{
		// Determine which side of the light is the point
		// minimum absolute value
		lightVec = normalize(worldPos - lightParams[fIndex].position.xyz);
		float maxVal = max(abs(lightVec.x), max(abs(lightVec.y), abs(lightVec.z)));
		vec3 axis = vec3(abs(lightVec.x) == maxVal ? 1.0f : 0.0f,
						 abs(lightVec.y) == maxVal ? 1.0f : 0.0f,
						 abs(lightVec.z) == maxVal ? 1.0f : 0.0f);
		vec3 lightVecSigns = sign(lightVec * axis);
		viewIndex = dot(abs(lightVecSigns), (abs((lightVecSigns - 1.0f) * 0.5f) + vec3(0.0f, 2.0f, 4.0f)));

		// Area light is half sphere
		if(lightParams[fIndex].position.w == GI_LIGHT_AREA)
			viewIndex = (lightVec.y < 0.0f) ? viewIndex : 2.0f;
	}
	else
	{
		// Determine Cascade
		float worldDist = max(0.0f, dot(worldPos - camPos.xyz, camDir.xyz));
	
		// Inv geom sum
		const float exponent = 1.2f;
		viewIndex = worldDist / camPos.w;
		viewIndex = floor(log2(viewIndex * (exponent - 1.0f) + 1.0f) / log2(exponent));
	}

	// Mult with proper cube side matrix
	vec4 clip = lightMatrices[fIndex].VPMatrices[uint(viewIndex)] * vec4(worldPos, 1.0f);

	// Convert to NDC
	vec3 ndc = clip.xyz / clip.w;

	// NDC to Tex
	float depth = 0.5f * ((2.0f * depthNearFar.x + 1.0f) + 
						(depthNearFar.y - depthNearFar.x) * ndc.z);

	if(lightParams[fIndex].position.w == GI_LIGHT_DIRECTIONAL)
		lightVec = vec3(0.5f * ndc.xy + 0.5f, viewIndex);

	return vec4(lightVec, depth);
}

float textureShadowLod(in sampler2DArray depths, 
					   in vec3 uv, 
					   in float lod, 
					   in float compareDepth)
{
    return step(compareDepth, textureLod(depths, uv, lod).r);
}

float textureShadowLodLerp(in sampler2DArray depths,
						   in vec2 size,
						   in vec3 uv, 
					       in float lod, 
						   in float compareDepth)
{
    vec2 texelSize = vec2(1.0) / size;
    vec2 f = fract(uv.xy * size + 0.5);
    vec2 centroidUV = floor(uv.xy * size + 0.5) / size;

    float lb = textureShadowLod(depths, vec3(centroidUV + texelSize * vec2(0.0, 0.0), uv.z), lod, compareDepth);
    float lt = textureShadowLod(depths, vec3(centroidUV + texelSize * vec2(0.0, 1.0), uv.z), lod, compareDepth);
    float rb = textureShadowLod(depths, vec3(centroidUV + texelSize * vec2(1.0, 0.0), uv.z), lod, compareDepth);
    float rt = textureShadowLod(depths, vec3(centroidUV + texelSize * vec2(1.0, 1.0), uv.z), lod, compareDepth);
    float a = mix(lb, lt, f.y);
    float b = mix(rb, rt, f.y);
    float c = mix(a, b, f.x);
    return c;
}

float ShadowSampleFlat(in vec4 shadowUV)
{
	float depth = 0.0f;
	if(lightParams[fIndex].position.w == GI_LIGHT_DIRECTIONAL)
	{
		vec3 uv = vec3(shadowUV.xy, float(fIndex * 6) + shadowUV.z);
		depth = textureLod(shadowMapsDir, uv, 0.0f).r;
	}
	else
	{
		depth = textureLod(shadowMaps, vec4(shadowUV.xyz, float(fIndex)), 0.0f).r;
	}
	return (depth < shadowUV.w) ? 0.0f : 1.0f;
}

float ShadowSample(in vec4 shadowUV)
{
	// Shadow new hier
	float shadowIntensity = 1.0f;
	for(uint i = 0; i < shadowMipCount; i++)
	{
		if(lightParams[fIndex].position.w == GI_LIGHT_DIRECTIONAL)
		{			
			vec3 uv = vec3(shadowUV.xy, float(fIndex * 6) + shadowUV.z);
			vec2 size = vec2(float(shadowMapWH >> i));
		
			float lodIntensity = 0.0f;
			for(int x = -1; x <= 1; x++){
			for(int y = -1; y <= 1; y++)
			{
				vec2 off = vec2(x,y)/size;
				lodIntensity += textureShadowLodLerp(shadowMapsDir, 
													 size, 
													 vec3(uv.xy + off, uv.z), 
													 i, 
													 shadowUV.w);
			}}
			lodIntensity /= 9.0f;
			lodIntensity = (1.0f - lodIntensity) * ((i == 0) ? 1.0f : 0.5f);
			//lodIntensity = (1.0f - lodIntensity);
			shadowIntensity -= lodIntensity / float(1 << i);
			
		}
		else
		{
			float depth = textureLod(shadowMaps, vec4(shadowUV.xyz, float(fIndex)), float(i)).x;
			shadowIntensity -= ((step(depth, shadowUV.w)) / (float(1 << i)));
		}
	}
	return max(0.0f, shadowIntensity);
}

vec3 PhongBDRF(in vec3 worldPos)
{
	vec3 lightIntensity = vec3(0.0f);

	// UV Coords
	vec2 gBuffUV = vec2(gl_FragCoord.xy - viewport.xy /*- vec2(0.5f)*/) / viewport.zw;
	vec4 shadowUV = CalculateShadowUV(worldPos);

	// Phong BDRF Calculation
	// Outputs intensity multiplier for each channel (rgb)
	// Diffuse is Lambert
	// We store normals in world space in GBuffer
	vec3 worldNormal = UnpackNormal(texture(gBuffNormal, gBuffUV).xy);
	vec3 worldEye = camPos.xyz - worldPos;
	
	// Light Vector and Light Falloff Calculation
	vec3 worldLight;
	float falloff = 1.0f;
	if(lightParams[fIndex].position.w == GI_LIGHT_DIRECTIONAL)
	{
		worldLight = -lightParams[fIndex].direction.xyz;
	}
	else
	{
		worldLight = lightParams[fIndex].position.xyz - worldPos;

		// Falloff Linear
		float lightRadius = lightParams[fIndex].direction.w;
		float distSqr = dot(worldLight.xyz, worldLight.xyz);

		// Linear Falloff
		//falloff = 1.0f - clamp(distSqr / (lightRadius * lightRadius), 0.0f, 1.0f);

		// Quadratic Falloff
		falloff = distSqr / (lightRadius * lightRadius);
		falloff = clamp(1.0f - falloff * falloff, 0.0f, 1.0f);
		falloff = (falloff * falloff) / (distSqr + 1.0f);
	}		
	worldLight = normalize(worldLight);
	worldNormal = normalize(worldNormal);
	worldEye = normalize(worldEye);
	vec3 worldReflect = normalize(-reflect(worldLight, worldNormal));
	vec3 worldHalf = normalize(worldLight + worldEye);

	// Diffuse Factor
	// Lambert Diffuse Model
	float lambertFactor = GI_ONE_OVER_PI * max(dot(worldNormal, worldLight), 0.0f);

	//// Burley Diffuse Model (Disney)
	//float rougness = 0.5f;
	//float NdL = dot(worldNormal, worldLight);
	//float NdV = dot(worldNormal, worldEye);
	//float LdH = max(dot(worldLight, worldHalf), 0.0f);
	//float fD90 = 0.5 + 2.0f * pow(LdH, 2.0f) * rougness;
	//lightIntensity = vec3(//(1.0f + (fD90 - 1.0f) * pow(1.0f - NdL, 5.0f)) *
	//					  //(1.0f + (fD90 - 1.0f) * pow(1.0f - NdV, 5.0f)) 
	//					  mix(1.0f, fD90, pow(clamp(1.0f - NdL, 0.0f, 1.0f), 5.0f)) *
	//					  mix(1.0f, fD90, pow(clamp(1.0f - NdV, 0.0f, 1.0f), 5.0f)) *
	//					  GI_ONE_OVER_PI);
	//lightIntensity = max(lightIntensity, vec3(0.0f));
	//lightIntensity *= NdL;

	// Early Bail From Light Occulusion
	// This also eliminates some self shadowing artifacts
	if(lambertFactor <= 0.0f) return vec3(0.0f);

	// Check Light Occulusion (ShadowMap)
	float shadowIntensity = ShadowSample(shadowUV);
	//float shadowIntensity = ShadowSampleFlat(shadowUV);
	
	// Early Bail from specular
	if(shadowIntensity <= 0.0f) return vec3(0.0f);

	////DEBUG	
	//// Cascade Check
	//if(lightParams[fIndex].position.w == GI_LIGHT_DIRECTIONAL)
	//{
	//	if(shadowUV.z == 0.0f)
	//		lightIntensity = vec3(1.0f, 0.0f, 0.0f);
	//	else if(shadowUV.z == 1.0f)
	//		lightIntensity = vec3(0.0f, 1.0f, 0.0f);
	//	else if(shadowUV.z == 2.0f)
	//		lightIntensity = vec3(0.0f, 0.0f, 1.0f);
	//	else if(shadowUV.z == 3.0f)
	//		lightIntensity = vec3(1.0f, 1.0f, 0.0f);
	//	else if(shadowUV.z == 4.0f)
	//		lightIntensity = vec3(1.0f, 0.0f, 1.0f);
	//	lightIntensity *= 0.1f;
	//}

	// Lambert
	lightIntensity = vec3(lambertFactor);

	// Specular
	float specPower = 32.0f + (texture(gBuffColor, gBuffUV).a) * 2048.0f;

	// Phong
	//lightIntensity += vec3(pow(max(dot(worldReflect, worldEye), 0.0f), specPower));
	// Blinn-Phong
	lightIntensity += vec3(pow(max(dot(worldHalf, worldNormal), 0.0f), specPower));

	// Falloff
	lightIntensity *= falloff;

	// Colorize
	lightIntensity *= lightParams[fIndex].color.rgb;

	// Intensity
	lightIntensity *= lightParams[fIndex].color.a;

	//Shadow
	lightIntensity *= shadowIntensity;

	// Out
	return lightIntensity;
}

void main(void)
{
	// Do Light Calculation
	vec3 lightIntensity = PhongBDRF(DepthToWorld());
	// Additive Blending will take care of the rest
	fboColor = vec4(lightIntensity, 1.0f);
}