#version 430
/*	
	**Voxel Trace Deferred Fetch Interpolated Compute Shader**
	
	File Name	: VoxTraceDeferredLerp.glsl
	Author		: Bora Yalciner
	Description	:

		Cuda does not support depth texture copy
		we need to copy depth values of the gbuffer to depth
*/

// Definitions
#define I_COLOR_FB layout(rgba8, binding = 2) restrict writeonly

#define LU_SVO_NODE layout(std430, binding = 2) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 3) readonly
#define LU_SVO_LEVEL_OFFSET layout(std430, binding = 4) readonly

#define U_RENDER_TYPE layout(location = 0)
#define U_FETCH_LEVEL layout(location = 1)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)

#define T_DEPTH layout(binding = 2)
#define T_DENSE_NODE layout(binding = 5)
#define T_DENSE_MAT layout(binding = 6)

#define FLT_MAX 3.402823466e+38F
#define EPSILON 0.00001f
#define SQRT_3 1.73205f

#define RENDER_TYPE_COLOR 0
#define RENDER_TYPE_OCCLUSION 1
#define RENDER_TYPE_NORMAL 2

// Uniforms
U_RENDER_TYPE uniform uint renderType;
U_FETCH_LEVEL uniform uint fetchLevel;

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

// Textures
uniform I_COLOR_FB image2D fbo;
uniform T_DEPTH sampler2D gBuffDepth;
uniform T_DENSE_NODE usampler3D tSVODense;
uniform T_DENSE_MAT usampler3D tSVOMat;

// Functions
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

					   in vec3 interpValue)
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

	colorA = mix(colorA, colorB, interpValue.x);
	colorB = mix(colorC, colorD, interpValue.x);
	colorC = mix(colorE, colorF, interpValue.x);
	colorD = mix(colorG, colorH, interpValue.x);

	colorA = mix(colorA, colorB, interpValue.y);
	colorB = mix(colorC, colorD, interpValue.y);

	color = mix(colorA, colorB, interpValue.z);
	
	vec4 normalA = UnpackNormalSVO(materialA.y);
	vec4 normalB = UnpackNormalSVO(materialB.y); 
	vec4 normalC = UnpackNormalSVO(materialC.y);
	vec4 normalD = UnpackNormalSVO(materialD.y); 
	vec4 normalE = UnpackNormalSVO(materialE.y); 
	vec4 normalF = UnpackNormalSVO(materialF.y); 
	vec4 normalG = UnpackNormalSVO(materialG.y); 
	vec4 normalH = UnpackNormalSVO(materialH.y);
		
	normalA = mix(normalA, normalB, interpValue.x);
	normalB = mix(normalC, normalD, interpValue.x);
	normalC = mix(normalE, normalF, interpValue.x);
	normalD = mix(normalG, normalH, interpValue.x);

	normalA = mix(normalA, normalB, interpValue.y);
	normalB = mix(normalC, normalD, interpValue.y);

	normal = mix(normalA, normalB, interpValue.z);

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

bool SampleSVO(out vec4 color,
			   out vec4 normal,
			   in vec3 worldPos)
{
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

	for(uint i = 0; i < 8; i++)
	{
		vec3 currentWorld = VoxPosToWorld(voxPosLevel + NEIG_MASK[i], fetchLevel);
		ivec3 voxPos = LevelVoxId(currentWorld, dimDepth.y);
	
		// Cull if out of bounds
		// Since cam is centered towards grid
		// Out of bounds means its cannot come towards the grid
		// directly cull
		if(any(lessThan(voxPos, ivec3(0))) ||
		   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
			continue;

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
			
			// .w component used to average so change it
			uvec2 mat = svoMaterial[loc].xy;
			if(traversedLevel == fetchLevel) mat.y |= 0xFF000000;

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
	}
	
	// Out
	InterpolateSparse(color, 
					  normal, 

					  matAB,
					  matCD,
					  matEF,
					  matGH,
							 
					  interp);

	if(normal.w == 0.0f) return false;
	return true;
}

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	if(any(greaterThanEqual(globalId, imageSize(fbo).xy))) return;

	// Fetch GBuffer and Interpolate Positions (if size is smaller than current gbuffer)
	vec2 gBuffUV = vec2(globalId + vec2(0.5f) - viewport.xy) / viewport.zw;
	vec3 worldPos = DepthToWorld(gBuffUV);

	vec4 color, normal;
	bool found = SampleSVO(color, normal, worldPos);

	vec3 outData = vec3(0.5f);
	if(found)
	{
		if(renderType == RENDER_TYPE_COLOR) 
			outData = color.xyz;
		else if(renderType == RENDER_TYPE_OCCLUSION)
			outData = vec3(1.0f - normal.w);
		else if(renderType == RENDER_TYPE_NORMAL)
			outData = normal.xyz;
	}
	imageStore(fbo, ivec2(globalId), vec4(outData, 0.0f));
}