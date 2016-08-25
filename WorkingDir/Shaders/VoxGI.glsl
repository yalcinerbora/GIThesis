#version 430
#extension GL_NV_shader_thread_group : require
#extension GL_NV_shader_thread_shuffle : require
/*	
	**Voxel Global Illumination Compute Shader**
	
	File Name	: VoxGI.glsl
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

#define U_LIGHT_INDEX layout(location = 2)
#define U_ON_OFF_SWITCH layout(location = 3)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)
#define U_CONE_PARAMS layout(std140, binding = 4)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)
#define T_DENSE_NODE layout(binding = 5)
#define T_DENSE_MAT layout(binding = 6)

#define CONE_COUNT 1

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define TRACE_NEIGBOUR 8
#define LOGICAL_CONES 1
#define NEIG 4

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f
#define GI_LIGHT_AREA 2.0f

#define GI_ONE_OVER_PI 0.318309f
#define PI 3.1415f

// Uniforms
U_LIGHT_INDEX uniform uint lIndex;
U_ON_OFF_SWITCH uniform uint specular;

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

	// x intensity AO
	// y intensity GI
	// z sqrt3
	// w falloffFactor
	vec4 coneParams2;
};

// Textures
uniform I_LIGHT_INENSITY image2D liTex;

uniform T_COLOR sampler2D gBuffColor;
uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;
uniform T_DENSE_NODE usampler3D tSVODense;
uniform T_DENSE_MAT usampler3D tSVOMat;

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

vec3 VoxPosToWorld(in ivec3  voxPos, in uint depth)
{
	return worldPosSpan.xyz + (vec3(voxPos) * (worldPosSpan.w * float(0x1 << (dimDepth.y - depth))));
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

bool InterpolateSparse(out vec4 color,
					   out vec4 normal,

					   in uvec4 matAB,
					   in uvec4 matCD,
					   in uvec4 matEF,
					   in uvec4 matGH,

					   in vec3 interpValue,
					   in bool specular)
{
	// Bigass fetch (its fast tho L1 cache doing work on GPU!!!)
	uvec2 materialA = matAB.xy;
	uvec2 materialB = matAB.zw;
	uvec2 materialC = matCD.xy;
	uvec2 materialD = matCD.zw;
	uvec2 materialE = matEF.xy;
	uvec2 materialF = matEF.zw;
	uvec2 materialG = matGH.xy;
	uvec2 materialH = matGH.zw;

	// Interp Color
	vec4 colorA = UnpackColorSVO(materialA.x);
	vec4 colorB = UnpackColorSVO(materialB.x); 
	vec4 colorC = UnpackColorSVO(materialC.x);
	vec4 colorD = UnpackColorSVO(materialD.x); 
	vec4 colorE = UnpackColorSVO(materialE.x); 
	vec4 colorF = UnpackColorSVO(materialF.x); 
	vec4 colorG = UnpackColorSVO(materialG.x); 
	vec4 colorH = UnpackColorSVO(materialH.x);

	if(TRACE_NEIGBOUR == 1 || specular)
	{
		color = colorA;
	}
	else
	{
		colorA = mix(colorA, colorB, interpValue.x);
		colorB = mix(colorC, colorD, interpValue.x);
		colorC = mix(colorE, colorF, interpValue.x);
		colorD = mix(colorG, colorH, interpValue.x);

		colorA = mix(colorA, colorB, interpValue.y);
		colorB = mix(colorC, colorD, interpValue.y);

		color = mix(colorA, colorB, interpValue.z);
	}
	
	vec4 normalA = UnpackNormalSVO(materialA.y);
	vec4 normalB = UnpackNormalSVO(materialB.y); 
	vec4 normalC = UnpackNormalSVO(materialC.y);
	vec4 normalD = UnpackNormalSVO(materialD.y); 
	vec4 normalE = UnpackNormalSVO(materialE.y); 
	vec4 normalF = UnpackNormalSVO(materialF.y); 
	vec4 normalG = UnpackNormalSVO(materialG.y); 
	vec4 normalH = UnpackNormalSVO(materialH.y);
		
	if(TRACE_NEIGBOUR == 1 || specular)
	{
		normal = normalA;
	}
	else
	{
		normalA = mix(normalA, normalB, interpValue.x);
		normalB = mix(normalC, normalD, interpValue.x);
		normalC = mix(normalE, normalF, interpValue.x);
		normalD = mix(normalG, normalH, interpValue.x);

		normalA = mix(normalA, normalB, interpValue.y);
		normalB = mix(normalC, normalD, interpValue.y);

		normal = mix(normalA, normalB, interpValue.z);
	}
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

void MatWrite(inout uvec4 matAB,
			  inout uvec4 matCD,
			  inout uvec4 matEF,
			  inout uvec4 matGH,
			  in uvec2 mat,
			  in uint i)
{
	if(i < 2)
	{
		matAB[(i % 2) * 2 + 0] = mat.x;
		matAB[(i % 2) * 2 + 1] = mat.y;
	}
	else if(i < 4)
	{
		matCD[(i % 2) * 2 + 0] = mat.x;
		matCD[(i % 2) * 2 + 1] = mat.y;
	}
	else if(i < 6)
	{
		matEF[(i % 2) * 2 + 0] = mat.x;
		matEF[(i % 2) * 2 + 1] = mat.y;
	}
	else if(i < 8)
	{
		matGH[(i % 2) * 2 + 0] = mat.x;
		matGH[(i % 2) * 2 + 1] = mat.y;
	}
}

// SVO Fetch
bool SampleSVO(out vec4 color,
			   out vec4 normal,
			   in vec3 worldPos,
			   in uint depth,
			   in bool specular)
{
	uint fetchLevel = depth;

	//	// Dense Fetch
	//if(fetchLevel <= dimDepth.w &&
	//   fetchLevel >= offsetCascade.w)
	//{
	//	uint mipId = dimDepth.w - fetchLevel;
	//	vec3 levelUV = LevelVoxIdF(worldPos, fetchLevel);			
	//	InterpolateDense(color, normal, levelUV, int(mipId));
	//	return true;
	//}

	//// For each Corner Value // Offsets
	//ivec3 voxPosLevel = LevelVoxId(worldPos, fetchLevel);
	//vec3 interp = LevelVoxIdF(worldPos, fetchLevel);
	//interp -= (vec3(voxPosLevel));
	//vec3 offsets = sign(interp);

	//// Materials that will be interpolated
	//uvec4 matAB = uvec4(0);
	//uvec4 matCD = uvec4(0);
	//uvec4 matEF = uvec4(0);
	//uvec4 matGH = uvec4(0);

	//for(uint i = 0; i < 8; i++)
	//{
	//	vec3 currentWorld = VoxPosToWorld(voxPosLevel + NEIG_MASK[i], fetchLevel);
	//	ivec3 voxPos = LevelVoxId(currentWorld, dimDepth.y);
	
	//	// Cull if out of bounds
	//	// Since cam is centered towards grid
	//	// Out of bounds means its cannot come towards the grid
	//	// directly cull
	//	if(any(lessThan(voxPos, ivec3(0))) ||
	//	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
	//		continue;

	//	// Initialize Traverse
	//	ivec3 denseVox = LevelVoxId(currentWorld, dimDepth.w);
	//	vec3 texCoord = vec3(denseVox) / dimDepth.z;
	//	unsigned int nodeIndex = texture(tSVODense, texCoord).x;
	//	unsigned int lastValid = nodeIndex;
	//	if(nodeIndex == 0xFFFFFFFF)
	//	{
	//		// Fall back to dense
	//		for(unsigned int j = 1; j < (dimDepth.w - offsetCascade.w); j++)
	//		{
	//			vec3 levelUV = LevelVoxIdF(worldPos, dimDepth.w - j);
	//			uvec2 mat = texelFetch(tSVOMat, ivec3(floor(levelUV)), int(j)).xy;

	//			if(mat.x != 0 || mat.y != 0)
	//			{
	//				MatWrite(matAB, matCD, matEF, matGH, mat, i);
	//				break;
	//			}
	//		}
	//		continue;
	//	}
	//	nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);
		
	//	// Tree Traverse
	//	uint traversedLevel;
	//	for(traversedLevel = dimDepth.w + 1; 
	//		traversedLevel < fetchLevel;
	//		traversedLevel++)
	//	{
	//		uint currentNode = svoNode[offsetCascade.y + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex];
	//		if(currentNode == 0xFFFFFFFF) break;
	//		lastValid = nodeIndex;
	//		nodeIndex = currentNode + CalculateLevelChildId(voxPos, traversedLevel + 1);
	//	}

	//	// Up until fetch
	//	uint loc = offsetCascade.z + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex;
	//	uvec2 mat = svoMaterial[loc].xy;

	//	// mat shouldnt be zero ever if it is allocated
	//	if((mat.x == 0x0 && mat.y == 0x0)) 
	//	{
	//		if(traversedLevel == (dimDepth.w + 1))
	//		{
	//			vec3 levelUV = LevelVoxIdF(worldPos, dimDepth.w);
	//			uvec2 mat = texelFetch(tSVOMat, ivec3(floor(levelUV)), 0).xy;
	//			MatWrite(matAB, matCD, matEF, matGH, mat, i);
	//			continue;
	//		}
	//		else
	//		{
	//			loc = offsetCascade.z + svoLevelOffset[traversedLevel - dimDepth.w - 1] + lastValid;
	//			mat = svoMaterial[loc].xy;
	//		}
	//	}
		
	//	if(traversedLevel == dimDepth.y) mat.y |= 0xFF000000;
	//	MatWrite(matAB, matCD, matEF, matGH, mat, i);
	//}
	
	//// Out
	//InterpolateSparse(color, 
	//				  normal, 

	//				  matAB,
	//				  matCD,
	//				  matEF,
	//				  matGH,
							 
	//				  interp,
	//				  specular);

	//if(normal.w == 0.0f) return false;
	//return true;

	// Dense Fetch
	if(fetchLevel <= dimDepth.w &&
	   fetchLevel >= offsetCascade.w)
	{
		uint mipId = dimDepth.w - fetchLevel;
		uint levelDim = dimDepth.z >> mipId;
		vec3 levelUV = LevelVoxIdF(worldPos, fetchLevel);
			
		InterpolateDense(color, normal, levelUV, int(mipId));
		return true;
	}

	// For each Corner Value // Offsets
	ivec3 voxPosLevel = LevelVoxId(worldPos, fetchLevel);
	vec3 interp = LevelVoxIdF(worldPos, fetchLevel);
	interp -= (vec3(voxPosLevel));
	vec3 offsets = sign(interp);

	// Materials that will be interpolated
	uvec4 matAB = uvec4(0);
	uvec4 matCD = uvec4(0);
	uvec4 matEF = uvec4(0);
	uvec4 matGH = uvec4(0);

	
	for(uint i = 0; i < ((specular) ? 1 : TRACE_NEIGBOUR); i++)
	{
		vec3 currentWorld = VoxPosToWorld(voxPosLevel + NEIG_MASK[i], fetchLevel);
		ivec3 voxPos = LevelVoxId(currentWorld, dimDepth.y);
	
		// Cull if out of bounds
		// Since cam is centered towards grid
		// Out of bounds means its cannot come towards the grid
		// directly cull
		if(any(lessThan(voxPos, ivec3(0))) ||
		   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
		{
			uvec2 mat = uvec2(0xFF99DAF0, 0xFF000000);
			MatWrite(matAB, matCD, matEF, matGH, mat, i);
			continue;
		}

		// Initialize Traverse
		unsigned int nodeIndex = 0;
		ivec3 denseVox = LevelVoxId(currentWorld, dimDepth.w);
		
		vec3 texCoord = vec3(denseVox) / dimDepth.z;
		nodeIndex = texture(tSVODense, texCoord).x;
		if(nodeIndex == 0xFFFFFFFF) continue;
		nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);
		
		// Tree Traverse
		uint traversedLevel;
		for(traversedLevel = dimDepth.w + 1; 
			traversedLevel < fetchLevel;
			traversedLevel++)
		{
			uint currentNode = svoNode[offsetCascade.y + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex];
			if(currentNode == 0xFFFFFFFF) break;
			nodeIndex = currentNode + CalculateLevelChildId(voxPos, traversedLevel + 1);
		}	

		// Mat out
		if(traversedLevel > (dimDepth.y - offsetCascade.x) || 
		   traversedLevel == fetchLevel)
		{
			// Mid or Leaf Level
			uint loc = offsetCascade.z + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex;
			uvec2 mat = svoMaterial[loc].xy;
			MatWrite(matAB, matCD, matEF, matGH, mat, i);			
		}
	}
	
	// Out
	InterpolateSparse(color, 
					  normal, 

					  matAB,
					  matCD,
					  matEF,
					  matGH,
							 
					  interp,
					  specular);

	if(normal.w == 0.0f) return false;
	return true;
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

vec3 IllumFactor(in vec3 coneDir,
			     in vec4 colorSVO,
    			 in vec4 normalSVO)
{
	float lightIntensity = 1.0f;

	// Light Intensity Relevant to the cone light angle (Lambert Factor)
	vec3 voxNormal = normalize(normalSVO.xyz);
	float lobeFactor = length(normalSVO.xyz);
	
	// Lambert Diffuse
	//lightIntensity *= max(dot(voxNormal.xyz, coneDir), 0.0f);

	// Sampled Lobe Factor
//	lightIntensity *= normalSVO.w;
//	lightIntensity *= lobeFactor;

	return lightIntensity * colorSVO.xyz * GI_ONE_OVER_PI;
	//return abs(voxNormal.xyz) * 2.0f;//0.005f;
}

vec3 CalculateConeDir(in vec3 ortho1, in vec3 ortho2, float angleRadian)
{
	return normalize(ortho1 * cos(angleRadian) + ortho2 * sin(angleRadian));
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId;

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
	

	uint dirId = ((pixelId.x  % (NEIG / 2)) * (NEIG / 2)) + (pixelId.y % (NEIG / 2));
	//uint dirId = pixelId.x  % NEIG;

	vec4 totals[LOGICAL_CONES + 1];
	for(uint i = 0; i < LOGICAL_CONES + 1; i++)
	{
		totals[i] = vec4(0.0f);
	}

	float surfaceOcclusion = 0.0f;
	vec3 surfaceIllumination = vec3(0.0f);
	uint coneCount = (specular == 0) ? CONE_COUNT : CONE_COUNT + 1;
	for(unsigned int i = 0; i < coneCount; i++)
	{
		// Previous surface point and occlusion data
		vec3 prevIllumination = vec3(0.0f);
		float prevOcclusion = 0.0f;
		float prevSurfPoint = 0.0f;

		// Initally Start the cone away from the surface since 
		// voxel system and polygon system are not %100 aligned
		vec3 initalTraceStart = worldPos + worldNorm * cascadeSpan * coneParams2.z * 2.0f;


		//// Find Corner points of the surface
		//float coneAperture = coneParams1.z;
		//vec3 coneDir;
		//if(i == CONE_COUNT)
		//{
		//	// Specular cone
		//	float specularity = texture(gBuffColor, gBuffUV).a;
		//	coneAperture = mix(coneParams1.z * 2.0f , coneParams1.z * 0.1f, specularity);

		//	// Find Corner points of the surface
		//	vec3 worldEye = normalize(camPos.xyz - worldPos);
		//	coneDir = normalize(-reflect(worldEye, worldNorm));
		//}
		//else
		//{			
		//	coneDir = normalize(worldNorm + coneAperture * CalculateConeDir(ortho1, ortho2, 
		//																	(i % LOGICAL_CONES) * (2.0f * PI / LOGICAL_CONES)));

		//	//coneDir = normalize(worldNorm + 
		//	//					ortho1 * coneParams1.z * CONE_ORTHO[i % 4].x + 
		//	//					ortho2 * coneParams1.z * CONE_ORTHO[i % 4].y);
		//}



		// Start sampling towards that direction
		// Loop Traverses until MaxDistance Exceeded
		// March distance is variable per iteration
		uint coneNo = 0;
		float marchDistance = cascadeSpan;
		for(float traversedDistance = cascadeSpan;
			traversedDistance <= coneParams1.x;
			traversedDistance += marchDistance)
		{
			// Find Corner points of the surface
			float coneAperture = coneParams1.z;
			vec3 coneDir;
			if(i == CONE_COUNT)
			{
				// Specular cone
				float specularity = texture(gBuffColor, gBuffUV).a;
				if(specularity < 0.5f) continue;
				coneAperture = mix(coneParams1.z, coneParams1.z * 0.3f, specularity);

				// Find Corner points of the surface
				vec3 worldEye = normalize(camPos.xyz - worldPos);
				coneDir = normalize(-reflect(worldEye, worldNorm));
			}
			else
			{			
				//coneDir = normalize(worldNorm + coneAperture * CalculateConeDir(ortho1, ortho2, 
				//																(coneNo % LOGICAL_CONES) * (2.0f * PI / LOGICAL_CONES)));

				coneAperture = tan(atan(coneAperture) * 1.5f);
				coneDir = normalize(worldNorm + coneAperture * CalculateConeDir(ortho1, ortho2, 
																				dirId * (2.0f * PI / NEIG)));

				//coneDir = normalize(worldNorm + 
				//					ortho1 * coneParams1.z * CONE_ORTHO[dirId].x + 
				//					ortho2 * coneParams1.z * CONE_ORTHO[dirId].y);
				//coneDir = normalize(worldNorm + 
				//					ortho1 * coneParams1.z * CONE_ORTHO[coneNo % 4].x + 
				//					ortho2 * coneParams1.z * CONE_ORTHO[coneNo % 4].y);
			}
			
			vec3 currentPos = initalTraceStart + coneDir * traversedDistance;

			// Current Cone Sample Diameter
			// and its corresponding depth and ratio
			float diameter = max(cascadeSpan, coneAperture * 2.0f * traversedDistance);
			uint nodeDepth = SpanToDepth(uint(floor(diameter / worldPosSpan.w)));
			//nodeDepth = 8;
		
			// Determine Coverage Span of the surface 
			// (wrt cone angle and distance from pixel)
			// And Store 3x3 voxels
			float surfacePoint = (traversedDistance + diameter * 0.5f);
				
			// start sampling from that surface (interpolate)
			vec4 color = vec4(0.0f), normal = vec4(0.0f);
			bool found = SampleSVO(color, normal, currentPos, nodeDepth, false/*i == CONE_COUNT*/);

			// Calculate Illumination & Occlusion
			float surfOcclusion = normal.w;
			vec3 illumSample = IllumFactor(coneDir, color, normal);
		
			// than interpolate with your previous surface's value to simulate quadlinear interpolation
			float ratio = (traversedDistance - prevSurfPoint) / (surfacePoint - prevSurfPoint);
//			float nodeOcclusion = mix(prevOcclusion, surfOcclusion, ratio);
//			vec3 illumination = mix(prevIllumination, illumSample, ratio);
			vec3 illumination = illumSample;
			float nodeOcclusion = surfOcclusion;
		
			// do AO calculations from this value (or values)
			// Correction Term to prevent intersecting samples error
			float diameterVoxelSize = worldPosSpan.w * (0x1 << (dimDepth.y - nodeDepth));
			nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, marchDistance / diameterVoxelSize);
			illumination = vec3(1.0f) - pow(vec3(1.0f) - illumination, vec3(marchDistance / diameterVoxelSize));
		
			// Occlusion falloff (linear)
			//nodeOcclusion *= (1.0f / (1.0f + coneParams2.w * diameter));
			nodeOcclusion *= (1.0f / (1.0f + coneParams2.w * traversedDistance));
			//nodeOcclusion *= (1.0f / (1.0f + pow(traversedDistance, 2.0f)));
			
			//illumination *= (1.0f / (1.0f + coneParams2.w * diameter));
			illumination *= (1.0f / (1.0f + coneParams2.w * traversedDistance));
			//illumination *= (1.0f / (1.0f + pow(traversedDistance, 2.0f)));

			// Average total occlusion value
			uint location = coneNo % LOGICAL_CONES;
			if(i == CONE_COUNT) location = LOGICAL_CONES;

			// Incorporation
			float factor = 1.0f;
			if(i != CONE_COUNT) factor = 2.0f;

			//totals[location].xyz += (vec3(1.0f) - totals[location].xyz) * illumination * dot(worldNorm, coneDir);
			totals[location].xyz += (1.0f - totals[location].w) * illumination /** dot(worldNorm, coneDir)*/ * factor;
			totals[location].w += (1.0f - totals[location].w) * nodeOcclusion * dot(worldNorm, coneDir);

			// Store Current Surface values as previous values
			prevSurfPoint = surfacePoint;
			prevOcclusion = surfOcclusion;
			prevIllumination = illumSample;

			// Advance sample point (from sampling diameter)
			marchDistance = diameter * coneParams1.w;
			coneNo++;
		}
		// Cos tetha multiplication
		//totalConeOcclusion *= dot(worldNorm, coneDir) * coneParams2.x;
		//totalIllumination *= dot(worldNorm, coneDir) * coneParams2.y;

		totals[0] *= (1.0f / LOGICAL_CONES);
		for(uint i = 1; i < LOGICAL_CONES; i++)
		{
			totals[0] += totals[i] * (1.0f / LOGICAL_CONES);
		}
		totals[0].xyz += totals[LOGICAL_CONES].xyz;
		totals[0].xyz *= coneParams2.y;
		totals[0].w *= coneParams2.x;
	}

	// Sum occlusion data
	vec4 result = vec4(totals[0].xyz, 1.0f - totals[0].w);
		
	// All Done!
	imageStore(liTex, ivec2(pixelId), result);
	//imageStore(liTex, ivec2(pixelId), vec4(result.w));
	//imageStore(liTex, ivec2(pixelId), vec4(worldPos, 0.0f));
	//imageStore(liTex, ivec2(pixelId), vec4(worldNorm, 0.0f));
}
