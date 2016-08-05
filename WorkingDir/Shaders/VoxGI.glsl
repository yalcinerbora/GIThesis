#version 430
#extension GL_NV_shader_thread_group : require
#extension GL_NV_shader_thread_shuffle : require
/*	
	**Voxel Global Illumination Compute Shader**
	
	File Name	: VoxAO.glsl
	Author		: Bora Yalciner
	Description	:

		Global Illumination approximation using SVO
*/

#define I_LIGHT_INENSITY layout(rgba8, binding = 2) restrict

#define LU_SVO_NODE layout(std430, binding = 2) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 3) readonly
#define LU_SVO_LEVEL_OFFSET layout(std430, binding = 4) readonly

#define LU_LIGHT layout(std430, binding = 1)
#define LU_LIGHT_MATRIX layout(std430, binding = 0)

#define U_SHADOW_MIP_COUNT layout(location = 0)
#define U_SHADOW_MAP_WH layout(location = 1)
#define U_LIGHT_INDEX layout(location = 2)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)
#define U_CONE_PARAMS layout(std140, binding = 4)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)
#define T_SHADOW layout(binding = 3)
#define T_SHADOW_DIR layout(binding = 4)
#define T_DENSE_NODE layout(binding = 5)
#define T_DENSE_MAT layout(binding = 6)

#define CONE_COUNT 4
#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f
#define GI_LIGHT_AREA 2.0f

#define GI_ONE_OVER_PI 0.318309f

// Uniforms
U_LIGHT_INDEX uniform uint lIndex;
U_SHADOW_MIP_COUNT uniform uint shadowMipCount;
U_SHADOW_MAP_WH uniform uint shadowMapWH;

uniform vec2 CONE_ORTHO[4] = 
{
	vec2( -1.0f, -1.0f),
    vec2( -1.0f, 1.0f),
    vec2( 1.0f, -1.0f),
    vec2( 1.0f, 1.0f)
};

uniform ivec3 NEIG_MASK[8] = 
{
	ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(0, 1, 0),
    ivec3(1, 1, 0),

	ivec3(0, 0, 1),
	ivec3(1, 0, 1),
	ivec3(0, 1, 1),
	ivec3(1, 1, 1)
};

LU_SVO_NODE buffer SVONode
{ 
	uint svoNode[];
};

LU_SVO_MATERIAL buffer SVOMaterial
{ 
	uvec2 svoMaterial[];
};

LU_SVO_LEVEL_OFFSET buffer SVOLevelOffsets
{
	uint svoLevelOffset[];
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

U_SVO_CONSTANTS uniform SVOConstants
{
	// xyz gridWorldPosition
	// w is gridSpan
	vec4 worldPosSpan;

	// x is grid dimension
	// y is grid depth
	// z is dense dimension
	// w is dense depth
	uvec4 dimDepth;

	// x is cascade count
	// y is node sparse offet
	// z is material sparse offset
	// w is dense mat tex min level
	uvec4 offsetCascade;
};

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

U_CONE_PARAMS uniform ConeTraceParameters
{
	// x max traverse distance
	// y tangent(coneAngle)
	// z tangent(coneAngle * 0.5f)
	// w sample ratio;
	vec4 coneParams1;

	// x intensity
	// y sqrt2
	// z sqrt3
	// w falloffFactor
	vec4 coneParams2;
};

// Textures
uniform I_LIGHT_INENSITY image2D liTex;

uniform T_COLOR sampler2D gBuffColor;
uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;
//uniform T_SHADOW samplerCubeArrayShadow shadowMaps;
//uniform T_SHADOW_DIR sampler2DArrayShadow shadowMapsDir;
uniform T_SHADOW samplerCubeArray shadowMaps;
uniform T_SHADOW_DIR sampler2DArray shadowMapsDir;
uniform T_DENSE_NODE usampler3D tSVODense;
uniform T_DENSE_MAT usampler3D tSVOMat;

// Surfaces traced by each pixel
shared uvec2 surface [BLOCK_SIZE_Y][(BLOCK_SIZE_X / CONE_COUNT) * (CONE_COUNT / 2)];

// Functions
vec3 DepthToWorld(vec2 gBuffUV)
{
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

ivec3 LevelVoxId(in vec3 worldPoint, in uint depth)
{
	ivec3 result = ivec3(floor((worldPoint - worldPosSpan.xyz) / worldPosSpan.w));
	return result >> (dimDepth.y - depth);
}

vec3 LevelVoxIdF(in vec3 worldPoint, in uint depth)
{
	return (worldPoint - worldPosSpan.xyz) / (worldPosSpan.w * float((0x1 << (dimDepth.y - depth))));
}

uint SpanToDepth(in uint number)
{
	return dimDepth.y - findMSB(number);
}

uint CalculateLevelChildId(in ivec3 voxPos, in uint levelDepth)
{
	uint bitSet = 0;
	bitSet |= ((voxPos.z >> (dimDepth.y - levelDepth)) & 0x000000001) << 2;
	bitSet |= ((voxPos.y >> (dimDepth.y - levelDepth)) & 0x000000001) << 1;
	bitSet |= ((voxPos.x >> (dimDepth.y - levelDepth)) & 0x000000001) << 0;
	return bitSet;
}

vec4 UnpackColorSVO(in uint colorPacked)
{
	return unpackUnorm4x8(colorPacked);
}

vec3 UnpackNormalGBuff(in uvec2 norm)
{
	vec3 result;
	result.x = ((float(norm.x) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float(norm.y & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm.y << 16));
	return result;
}

vec4 UnpackNormalSVO(in uint voxNormPosY)
{
	return vec4(unpackSnorm4x8(voxNormPosY).xyz,
		        unpackUnorm4x8(voxNormPosY).w);
}

vec4 CalculateShadowUV(in vec3 worldPos, in uint lightIndex)
{
	float viewIndex = 0.0f;
	vec3 lightVec;
	if(lightParams[lightIndex].position.w == GI_LIGHT_POINT ||
		lightParams[lightIndex].position.w == GI_LIGHT_AREA)
	{
		// Determine which side of the light is the point
		// minimum absolute value
		lightVec = worldPos - lightParams[lightIndex].position.xyz;
		float maxVal = max(abs(lightVec.x), max(abs(lightVec.y), abs(lightVec.z)));
		vec3 axis = vec3(abs(lightVec.x) == maxVal ? 1.0f : 0.0f,
						 abs(lightVec.y) == maxVal ? 1.0f : 0.0f,
						 abs(lightVec.z) == maxVal ? 1.0f : 0.0f);
		vec3 lightVecSigns = sign(lightVec * axis);
		viewIndex = dot(abs(lightVecSigns), (abs((lightVecSigns - 1.0f) * 0.5f) + vec3(0.0f, 2.0f, 4.0f)));

		// Area light is half sphere
		if(lightParams[lightIndex].position.w == GI_LIGHT_AREA)
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
	vec4 clip = lightMatrices[lightIndex].VPMatrices[uint(viewIndex)] * vec4(worldPos, 1.0f);

	// Convert to NDC
	vec3 ndc = clip.xyz / clip.w;

	// NDC to Tex
	float depth = 0.5 * ((2.0f * depthNearFar.x + 1.0f) + 
						(depthNearFar.y - depthNearFar.x) * ndc.z);

	if(lightParams[lightIndex].position.w == GI_LIGHT_DIRECTIONAL)
		lightVec = vec3(0.5f * ndc.xy + 0.5f, viewIndex);

	return vec4(lightVec, depth);
}

bool InterpolateSparse(out vec4 color,
					   out vec4 normal,

					   in vec3 worldPos,
					   in uint depth,
					   in uint matLoc)
{
	ivec3 voxPosLevel = LevelVoxId(worldPos, depth - 1);
	vec3 voxPosWorld = worldPosSpan.xyz + vec3(voxPosLevel) * (worldPosSpan.w * (0x1 << dimDepth.y - (depth - 1)));
	vec3 interpValue = (worldPos - voxPosWorld) / (worldPosSpan.w * (0x1 << dimDepth.y - (depth - 1)));

	// Bigass fetch (its fast tho L1 cache doing work on GPU!!!)
	uvec2 materialA = svoMaterial[matLoc + 0].xy;
	//uvec2 materialB = svoMaterial[matLoc + 1].xy;
	//uvec2 materialC = svoMaterial[matLoc + 2].xy;
	//uvec2 materialD = svoMaterial[matLoc + 3].xy;
	//uvec2 materialE = svoMaterial[matLoc + 4].xy;
	//uvec2 materialF = svoMaterial[matLoc + 5].xy;
	//uvec2 materialG = svoMaterial[matLoc + 6].xy;
	//uvec2 materialH = svoMaterial[matLoc + 7].xy;

	// Interp Color
	vec4 colorA = UnpackColorSVO(materialA.x);
	//vec4 colorB = UnpackColorSVO(materialB.x); 
	//vec4 colorC = UnpackColorSVO(materialC.x);
	//vec4 colorD = UnpackColorSVO(materialD.x); 
	//vec4 colorE = UnpackColorSVO(materialE.x); 
	//vec4 colorF = UnpackColorSVO(materialF.x); 
	//vec4 colorG = UnpackColorSVO(materialG.x); 
	//vec4 colorH = UnpackColorSVO(materialH.x);

	//colorA = mix(colorA, colorB, interpValue.x);
	//colorB = mix(colorC, colorD, interpValue.x);
	//colorC = mix(colorE, colorF, interpValue.x);
	//colorD = mix(colorG, colorH, interpValue.x);

	//colorA = mix(colorA, colorB, interpValue.y);
	//colorB = mix(colorC, colorD, interpValue.y);

	//color = mix(colorA, colorB, interpValue.z);

	color = colorA;
	
	vec4 normalA = UnpackNormalSVO(materialA.y);
	//vec4 normalB = UnpackNormalSVO(materialB.y); 
	//vec4 normalC = UnpackNormalSVO(materialC.y);
	//vec4 normalD = UnpackNormalSVO(materialD.y); 
	//vec4 normalE = UnpackNormalSVO(materialE.y); 
	//vec4 normalF = UnpackNormalSVO(materialF.y); 
	//vec4 normalG = UnpackNormalSVO(materialG.y); 
	//vec4 normalH = UnpackNormalSVO(materialH.y);
	
	// .w component used to average so change it
	if(depth == dimDepth.y)
	{
		normalA.w = ceil(normalA.w);
		//normalB.w = ceil(normalB.w);
		//normalC.w = ceil(normalC.w);
		//normalD.w = ceil(normalD.w);
		//normalE.w = ceil(normalE.w);
		//normalF.w = ceil(normalF.w);
		//normalG.w = ceil(normalG.w);
		//normalH.w = ceil(normalH.w);
	}
	
	//normalA = mix(normalA, normalB, interpValue.x);
	//normalB = mix(normalC, normalD, interpValue.x);
	//normalC = mix(normalE, normalF, interpValue.x);
	//normalD = mix(normalG, normalH, interpValue.x);

	//normalA = mix(normalA, normalB, interpValue.y);
	//normalB = mix(normalC, normalD, interpValue.y);

	//normal = mix(normalA, normalB, interpValue.z);

	normal = normalA;

	if(normal.w == 0.0f) return false;
	return true;
}

void InterpolateDense(out vec4 color,
					   out vec4 normal,
					
					   in vec3 levelUV, 
					   in int level)
{
	vec3 interpolId = levelUV - floor(levelUV);
	ivec3 uvInt = ivec3(floor(levelUV));

	uvec2 materialA = texelFetch(tSVOMat, uvInt + NEIG_MASK[0], level).xy;
	uvec2 materialB = texelFetch(tSVOMat, uvInt + NEIG_MASK[1], level).xy;
	uvec2 materialC = texelFetch(tSVOMat, uvInt + NEIG_MASK[2], level).xy;
	uvec2 materialD = texelFetch(tSVOMat, uvInt + NEIG_MASK[3], level).xy;
	uvec2 materialE = texelFetch(tSVOMat, uvInt + NEIG_MASK[4], level).xy;
	uvec2 materialF = texelFetch(tSVOMat, uvInt + NEIG_MASK[5], level).xy;
	uvec2 materialG = texelFetch(tSVOMat, uvInt + NEIG_MASK[6], level).xy;
	uvec2 materialH = texelFetch(tSVOMat, uvInt + NEIG_MASK[7], level).xy;

	vec4 colorA = UnpackColorSVO(materialA.x);
	vec4 colorB = UnpackColorSVO(materialB.x);
	vec4 colorC = UnpackColorSVO(materialC.x);
	vec4 colorD = UnpackColorSVO(materialD.x);
	vec4 colorE = UnpackColorSVO(materialE.x);
	vec4 colorF = UnpackColorSVO(materialF.x);
	vec4 colorG = UnpackColorSVO(materialG.x);
	vec4 colorH = UnpackColorSVO(materialH.x);
	
	colorA = mix(colorA, colorB, interpolId.x);
	colorB = mix(colorC, colorD, interpolId.x);
	colorC = mix(colorE, colorF, interpolId.x);
	colorD = mix(colorG, colorH, interpolId.x);

	colorA = mix(colorA, colorB, interpolId.y);
	colorB = mix(colorC, colorD, interpolId.y);

	color = mix(colorA, colorB, interpolId.z);
	//color = colorA;

	vec4 normalA = UnpackNormalSVO(materialA.y);
	vec4 normalB = UnpackNormalSVO(materialB.y);
	vec4 normalC = UnpackNormalSVO(materialC.y);
	vec4 normalD = UnpackNormalSVO(materialD.y);
	vec4 normalE = UnpackNormalSVO(materialE.y);
	vec4 normalF = UnpackNormalSVO(materialF.y);
	vec4 normalG = UnpackNormalSVO(materialG.y);
	vec4 normalH = UnpackNormalSVO(materialH.y);

	normalA = mix(normalA, normalB, interpolId.x);
	normalB = mix(normalC, normalD, interpolId.x);
	normalC = mix(normalE, normalF, interpolId.x);
	normalD = mix(normalG, normalH, interpolId.x);

	normalA = mix(normalA, normalB, interpolId.y);
	normalB = mix(normalC, normalD, interpolId.y);

	normal = mix(normalA, normalB, interpolId.z);
	//normal = normalA;
}

// SVO Fetch
bool SampleSVO(out vec4 color,
			   out vec4 normal,
			   in vec3 worldPos,
			   in uint depth)
{
	ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);
	
	// Cull if out of bounds
	// Since cam is centered towards grid
	// Out of bounds means its cannot come towards the grid
	// directly cull
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
		return false;

	// Dense Fetch
	if(depth <= dimDepth.w &&
	   depth >= offsetCascade.w)
	{
		uint mipId = dimDepth.w - depth;
		uint levelDim = dimDepth.z >> mipId;
		vec3 levelUV = LevelVoxIdF(worldPos, depth);
			
		InterpolateDense(color, normal, levelUV, int(mipId));
		return true;
	}

	// Initialize Traverse
	unsigned int nodeIndex = 0;
	ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
	vec3 texCoord = vec3(denseVox) / dimDepth.z;
	nodeIndex = texture(tSVODense, texCoord).x;
	if(nodeIndex == 0xFFFFFFFF) return false;
	nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);

	// Tree Traverse
	uint traversedLevel;
	for(traversedLevel = dimDepth.w + 1; 
		traversedLevel < depth;
		traversedLevel++)
	{
		uint currentNode = svoNode[offsetCascade.y + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex];
		if(currentNode == 0xFFFFFFFF) return false;//break;
		nodeIndex = currentNode + CalculateLevelChildId(voxPos, traversedLevel + 1);
	}
	//nodeIndex -= CalculateLevelChildId(voxPos, traversedLevel);

	// Mat out
	if(traversedLevel > (dimDepth.y - offsetCascade.x) || 
	   traversedLevel == depth)
	{
		// Mid or Leaf Level
		uint loc = offsetCascade.z + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex;
		return InterpolateSparse(color, normal, worldPos, traversedLevel, loc);
	}
	return false;
}

void SumPixelData(inout vec4 coneColorOcc)
{
	for(int offset = (CONE_COUNT / 2); offset > 0; offset /= 2)
	{
		vec4 neigNode = shuffleDownNV(coneColorOcc, offset, CONE_COUNT);
		coneColorOcc = mix(coneColorOcc, neigNode, 0.5f);
		//coneColorOcc += neigNode;
	}
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

float ShadowSample(in vec4 shadowUV, in uint shadowLod, in uint lightIndex)
{
	// Shadow new hier
	float shadowIntensity = 1.0f;
//	for(uint i = 0; i < shadowMipCount; i++)
//	{
		if(lightParams[lightIndex].position.w == GI_LIGHT_DIRECTIONAL)
		{			
			vec3 uv = vec3(shadowUV.xy, float(lightIndex * 6) + shadowUV.z);
			vec2 size = vec2(float(shadowMapWH >> shadowLod));
		
			float lodIntensity = 0.0f;
			for(int x = -1; x <= 1; x++){
			for(int y = -1; y <= 1; y++)
			{
				vec2 off = vec2(x,y)/size;
				lodIntensity += textureShadowLodLerp(shadowMapsDir, 
													 size, 
													 vec3(uv.xy + off, uv.z), 
													 shadowLod, 
													 shadowUV.w);
			}}
			lodIntensity /= 9.0f;
			lodIntensity = (1.0f - lodIntensity) * ((shadowLod == 0) ? 1.0f : 0.5f);
			//lodIntensity = (1.0f - lodIntensity);
			shadowIntensity -= lodIntensity / float(1 << shadowLod);
			
		}
		else
		{
			float depth = textureLod(shadowMaps, vec4(shadowUV.xyz, float(lightIndex)), float(shadowLod)).x;
			shadowIntensity -= ((step(depth, shadowUV.w)) / (float(1 << shadowLod)));
		}
//	}
	return max(0.0f, shadowIntensity);
}

vec3 PhongBRDF(in vec3 worldPos,
			   in vec4 colorSVO,
			   in vec4 normalSVO,
			   in uint lightIndex,
			   in uint nodeDepth)
{
	vec3 lightIntensity = vec3(0.0f);

	vec3 worldEye = camPos.xyz - worldPos;
	vec4 shadowUV = CalculateShadowUV(worldPos, lightIndex);
	vec3 worldNormal = normalize(normalSVO.xyz);
	float normalAccuracy = length(normalSVO.xyz);

	vec3 worldLight;
	float falloff = 1.0f;
	if(lightParams[lightIndex].position.w == GI_LIGHT_DIRECTIONAL)
	{
		worldLight = -lightParams[lightIndex].direction.xyz;
	}
	else
	{
		worldLight = lightParams[lightIndex].position.xyz - worldPos;

		// Falloff Linear
		float lightRadius = lightParams[lightIndex].direction.w;
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
	vec3 worldHalf = normalize(worldLight + worldEye);

	// Lambert Diffuse Model
	lightIntensity = GI_ONE_OVER_PI * vec3(max(dot(worldNormal, worldLight), 0.0f));

	// Early Bail From Light Occulusion
	// This also eliminates some self shadowing artifacts
	if(lightIntensity == vec3(0.0f))
		return vec3(0.0f);

	// Check Light Occulusion (ShadowMap)
	// TODO: Check projected pixel sizes and choose the most appropirate level on the
	// Hierarhical shadow map
	uint shadowLod = max(3, dimDepth.y - nodeDepth);
	float shadowIntensity = ShadowSample(shadowUV, shadowLod, lightIndex);

	// Specular
	float specPower = colorSVO.w * 4096.0f;

	// Blinn-Phong
	lightIntensity += vec3(pow(max(dot(worldHalf, worldNormal), 0.0f), specPower));

	// Sampled Normal Accuracy
	lightIntensity *= normalAccuracy;
	
	// Falloff
	lightIntensity *= falloff;

	// Colorize
	lightIntensity *= lightParams[lightIndex].color.rgb;

	// Intensity
	lightIntensity *= lightParams[lightIndex].color.a;

	//Shadow
	lightIntensity *= shadowIntensity;

	// Voxel Occlusion
//	lightIntensity *= normalSVO.w;

	// Out
	return lightIntensity * colorSVO.xyz;
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId / uvec2(CONE_COUNT, 1);

	if(any(greaterThanEqual(pixelId, imageSize(liTex).xy))) return;

	// Fetch GBuffer and Interpolate Positions (if size is smaller than current gbuffer)
	vec2 gBuffUV = vec2(pixelId + vec2(0.5f) - viewport.xy) / viewport.zw;
	vec3 worldPos = DepthToWorld(gBuffUV);
	vec3 worldNorm = UnpackNormalGBuff(texture(gBuffNormal, gBuffUV).xy);

	// Determine cascade no from distance of the camera
	// And the min span for that cascade
	vec3 gridCenter = worldPosSpan.xyz + worldPosSpan.w * (dimDepth.x >> 1);
	vec3 diff = abs(worldPos - gridCenter) / (worldPosSpan.w * (dimDepth.x >> offsetCascade.x));
	uvec3 cascades = findMSB(uvec3(diff)) + 1;
	uint cascadeNo = uint(max(cascades.x, max(cascades.y, cascades.z)));
	float cascadeSpan = worldPosSpan.w * (0x1 << cascadeNo);
	
	// Find Edge vectors from normal
	// [(-z-y) / x, 1, 1] is perpendicular (unless normal is X axis)
	// handle special case where normal is (1.0f, 0.0f, 0.0f)
	vec3 ortho1 = normalize(vec3(-(worldNorm.z + worldNorm.y) / worldNorm.x, 1.0f, 1.0f));
	ortho1 = mix(ortho1, vec3(0.0f, 1.0f, 0.0f), floor(worldNorm.x));
	vec3 ortho2 = normalize(cross(worldNorm, ortho1));

	// Find Corner points of the surface
	vec3 coneDir = normalize(worldNorm + 
							 ortho1 * coneParams1.z * CONE_ORTHO[globalId.x % CONE_COUNT].x + 
							 ortho2 * coneParams1.z * CONE_ORTHO[globalId.x % CONE_COUNT].y);

	// Previous surface point and occlusion data
	float totalConeOcclusion = 0.0f;
	vec3 totalIllumination = vec3(0.0f);
	float prevOcclusion = 0.0f;
	float prevSurfPoint = 0.0f;

	// Initally Start the cone away from the surface since 
	// voxel system and polygon system are not %100 aligned
	worldPos += coneDir * cascadeSpan * coneParams2.z * 2;

	// Start sampling towards that direction
	// Loop Traverses until MaxDistance Exceeded
	// March distance is variable per iteration
	float marchDistance = cascadeSpan;
	for(float traversedDistance = cascadeSpan;
		traversedDistance <= coneParams1.x;
		traversedDistance += marchDistance)
	{
		vec3 currentPos = worldPos + coneDir * traversedDistance;

		// Current Cone Sample Diameter
		// and its corresponding depth
		float diameter = max(cascadeSpan, coneParams1.z * 2.0f * traversedDistance);
		uint nodeDepth = SpanToDepth(uint(round(diameter / worldPosSpan.w)));
		//nodeDepth = 7;
		
		// Determine Coverage Span of the surface 
		// (wrt cone angle and distance from pixel)
		// And Store 3x3 voxels
		float surfacePoint = (traversedDistance + diameter * 0.5f);
				
		// start sampling from that surface (interpolate)
		vec4 color, normal;// == Garbage comes from here
		bool found = SampleSVO(color, normal, currentPos, nodeDepth);

		// Calculate Illumination & Occlusion
		float surfOcclusion = (found) ? normal.w : 0.0f;
		vec3 illumSample = (found) ? PhongBRDF(currentPos, color, normal, lIndex, nodeDepth) : vec3(0.0f);
		
		// Omit if %100 occuluded in closer ranges
		// Since its not always depth pos aligned with voxel pos
//		bool isOmitDistance = (surfOcclusion > 0.9f) && (traversedDistance < (coneParams2.z * worldPosSpan.w * (0x1 << offsetCascade.x - 1)));
//		surfOcclusion = isOmitDistance ? 0.0f : surfOcclusion;		

		// than interpolate with your previous surface's value to simulate quadlinear interpolation
		float ratio = (traversedDistance - prevSurfPoint) / (surfacePoint - prevSurfPoint);
		float nodeOcclusion = mix(prevOcclusion, surfOcclusion, ratio);
		
		// do AO calculations from this value (or values)
		// Correction Term to prevent intersecting samples error
		float diameterVoxelSize = worldPosSpan.w * (0x1 << (dimDepth.y - nodeDepth));
		nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, marchDistance / diameterVoxelSize);
		illumSample = vec3(1.0f) - pow(vec3(1.0f) - illumSample, vec3(marchDistance / diameterVoxelSize));
		
		// Occlusion falloff (linear)
		nodeOcclusion *= (1.0f / (1.0f + coneParams2.w * diameter));
		illumSample *= (1.0f / (1.0f + coneParams2.w * diameter));
		//illumSample *= (1.0f / (1.0f + coneParams2.w * traversedDistance));
		//nodeOcclusion *= (1.0f / (1.0f + coneParams2.w * traversedDistance);
		//illumSample *= (1.0f / (1.0f + pow(traversedDistance, 0.2f)));
		//nodeOcclusion *= (1.0f / (1.0f + pow(traversedDistance, 0.2f)));

		// Average total occlusion value
		totalConeOcclusion += (1.0f - totalConeOcclusion) * nodeOcclusion;
		totalIllumination += (vec3(1.0f) - totalIllumination) * illumSample;

		// Store Current Surface values as previous values
		prevOcclusion = surfOcclusion;
		prevSurfPoint = surfacePoint;

		// Advance sample point (from sampling diameter)
		marchDistance = diameter * coneParams1.w;
	}
	// Cos tetha multiplication
	totalConeOcclusion *= dot(worldNorm, coneDir) * coneParams2.x;
	totalIllumination *= dot(worldNorm, coneDir) * coneParams2.x;

	// Sum occlusion data
	vec4 result = vec4(totalIllumination, 1.0f - totalConeOcclusion);
	SumPixelData(result);
	
	// All Done!
	if(globalId.x % CONE_COUNT == 0)
	{
		imageStore(liTex, ivec2(pixelId), result);
		//imageStore(liTex, ivec2(pixelId), vec4(result.w));
		//imageStore(liTex, ivec2(pixelId), vec4(worldPos, 0.0f));
		//imageStore(liTex, ivec2(pixelId), vec4(worldNorm, 0.0f));
	}
}
