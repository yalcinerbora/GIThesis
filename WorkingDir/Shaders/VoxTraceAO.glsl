#version 430
/*	
	**Voxel Ambient Occulusion Compute Shader**
	
	File Name	: VoxTraceAO.vert
	Author		: Bora Yalciner
	Description	:

		Ambient Occulusion approximation using SVO
*/

#define I_LIGHT_INENSITY layout(rgba8, binding = 2) restrict writeonly

#define LU_SVO_NODE layout(std430, binding = 0) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 1) readonly
#define LU_SVO_LEVEL_OFFSET layout(std430, binding = 2) readonly

#define U_MAX_DISTANCE layout(location = 0)
#define U_CONE_ANGLE layout(location = 1)
#define U_SAMPLE_DISTANCE layout(location = 2)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)

#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Static cone count for faster implementation (prob i'll switch shaders instead of dynamically writing it)
#define CONE_COUNT 4		// Total cone count
#define TRACE_RATIO 1
#define SQRT3 1.732f

uniform vec2 CONE_ID_MAP[ 4 ] = 
{
	vec2( 0.0f, 0.0f),
    vec2( 0.0f, 1.0f),
    vec2( 1.0f, 0.0f),
    vec2( 1.0f, 1.0f)
};

U_CONE_ANGLE uniform float coneAngle;
U_MAX_DISTANCE uniform float maxDistance;
U_SAMPLE_DISTANCE uniform float sampleDistanceRatio;

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
	// w is renderLevel
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
uniform I_LIGHT_INENSITY image2D liTex;

uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;

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

vec3 UnpackColor(in uint colorPacked)
{
	vec3 color;
	color.x = float((colorPacked & 0x000000FF) >> 0) / 255.0f;
	color.y = float((colorPacked & 0x0000FF00) >> 8) / 255.0f;
	color.z = float((colorPacked & 0x00FF0000) >> 16) / 255.0f;
	return color;
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

vec3 UnpackNormalSVO(in uint voxNormPosY)
{
	vec3 result;
	result.x = ((float(voxNormPosY & 0xFFFF) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float((voxNormPosY >> 16) & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(voxNormPosY));
	
	return result;
}

float UnpackOcculusion(in uint colorPacked)
{
	return float((colorPacked & 0xFF000000) >> 24) / 255.0f;
}

vec3 InterpolatePos(in vec3 worldPos)
{
	// Interpolate position if gBufferTex > traceTex
	if(TRACE_RATIO == 1) return worldPos;
	else
	{
		// TODO: Implement
		// Use sibling cone threads and shared memory to reduce neigbouring pixels
		// dimensional difference has to be power of two
		return worldPos;
	}
}

vec3 InterpolateNormal(in vec3 worldNormal)
{
	
	if(TRACE_RATIO == 1) return worldNormal;
	else
	{
		// TODO: Implement
		// Use sibling cone threads and shared memory to reduce neigbouring pixels
		// dimensional difference has to be power of two
		return worldNormal;
	}
}

float TripolateOcclusion(in vec3 worldPos,
						 in uint depth,
						 in uint matLoc)
{
	// Bigass fetch (its fast tho L1 cache doing work! on GPU!!!)
	vec4 first, last;
	first.x = UnpackOcculusion(svoMaterial[matLoc + 0].x);
	first.y = UnpackOcculusion(svoMaterial[matLoc + 1].x);
	first.z = UnpackOcculusion(svoMaterial[matLoc + 2].x);
	first.w = UnpackOcculusion(svoMaterial[matLoc + 3].x);

	last.x = UnpackOcculusion(svoMaterial[matLoc + 4].x);
	last.y = UnpackOcculusion(svoMaterial[matLoc + 5].x);
	last.z = UnpackOcculusion(svoMaterial[matLoc + 6].x);
	last.w = UnpackOcculusion(svoMaterial[matLoc + 7].x);

	// Last level AO value is invalid (it used as avg count)
	if(depth == dimDepth.y)
	{
		first = ceil(first);
		last = ceil(last);
	}

	ivec3 voxPosLevel = LevelVoxId(worldPos, depth - 1);
	vec3 voxPosWorld = worldPosSpan.xyz + vec3(voxPosLevel) * (worldPosSpan.w * (0x1 << dimDepth.y - (depth - 1)));
	voxPosWorld = (worldPos - voxPosWorld) / (worldPosSpan.w * (0x1 << dimDepth.y - (depth - 1)));

	vec4 lerpBuff;
	lerpBuff.x = mix(first.x, first.y, voxPosWorld.x);
	lerpBuff.y = mix(first.z, first.w, voxPosWorld.x);
	lerpBuff.z = mix(last.x, last.y, voxPosWorld.x);
	lerpBuff.w = mix(last.z, last.w, voxPosWorld.x);

	lerpBuff.x = mix(lerpBuff.x, lerpBuff.y, voxPosWorld.y);
	lerpBuff.y = mix(lerpBuff.z, lerpBuff.w, voxPosWorld.y);

	lerpBuff.x = mix(lerpBuff.x, lerpBuff.y, voxPosWorld.z);
	return lerpBuff.x;
}

float SampleSVOOcclusion(in vec3 worldPos, in uint depth)
{
	// Start tracing (stateless start from root (dense))
	ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);

	// Cull if out of bounds
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
		return 0;

	// Tripolation is different if its sparse or dense
	if(depth <= dimDepth.w)
	{
		// TODO:
		return 0;
	}
	else
	{
		ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
		uint nodeIndex = svoNode[denseVox.z * dimDepth.z * dimDepth.z +
								 denseVox.y * dimDepth.z + 
								 denseVox.x];
		if(nodeIndex == 0xFFFFFFFF) return 0;
		nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);
		for(uint i = dimDepth.w + 1; i < depth; i++)
		{
			// Fetch Next Level
			uint newNodeIndex = svoNode[offsetCascade.y +
										svoLevelOffset[i - dimDepth.w] +
										nodeIndex];

			// Node check (Empty node also not leaf)
			// Means object does not
			if(newNodeIndex == 0xFFFFFFFF)
			{
				//nodeIndex -= CalculateLevelChildId(voxPos, i);
				//uint matLoc = offsetCascade.z + svoLevelOffset[i - dimDepth.w] + nodeIndex;
				//return TripolateOcclusion(worldPos, i, matLoc);
				return 0;
			}
			else
			{
				// Node has value
				// Go deeper
				nodeIndex = newNodeIndex + CalculateLevelChildId(voxPos, i + 1);
			}
		}
		// Finally At requested level
		// BackTrack From Child
		nodeIndex -= CalculateLevelChildId(voxPos, depth);
		uint matLoc = offsetCascade.z + svoLevelOffset[depth - dimDepth.w] +
					  nodeIndex;
		return TripolateOcclusion(worldPos, depth, matLoc); 
	}
}


// Shared Mem
shared float reduceBuffer[BLOCK_SIZE_Y * (BLOCK_SIZE_X / CONE_COUNT)][(CONE_COUNT / 2)]; 
void SumPixelOcclusion(inout float totalConeOcclusion)
{
	uvec2 localId = gl_LocalInvocationID.xy;
	uvec2 sMemId = uvec2(localId.y * (BLOCK_SIZE_X / CONE_COUNT) + (localId.x / CONE_COUNT),
						 localId.x % (CONE_COUNT / 2));

	uint pixelConeId = localId.x % CONE_COUNT;

	// left ones share their data
	if(pixelConeId >= 2) 
		reduceBuffer[sMemId.x][sMemId.y] = totalConeOcclusion;
	memoryBarrierShared();

	// right ones reduce
	if(pixelConeId < 2) 
	{
		// Lerp it at the middle (weighted avg)
		totalConeOcclusion += reduceBuffer[sMemId.x][sMemId.y];
		if(pixelConeId == 1) reduceBuffer[sMemId.x][0] = totalConeOcclusion;
	}
	memoryBarrierShared();
	if(pixelConeId == 0) totalConeOcclusion += reduceBuffer[sMemId.x][0];
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
	worldPos = InterpolatePos(worldPos); 
	worldNorm = InterpolateNormal(worldNorm);

	// Align voxel Space and World Space (voxel space incremented by multiples of grid span)
	// (so its slightly shifted no need to convert via matrix mult)
	//vec3 dif = (worldPosSpan.xyz + (dimDepth.x * 0.5f * worldPosSpan.w)) - camPos.xyz;
	//worldPos -= dif;

	// Each Thread Has locally same location now generate cones
	// We will cast 4 Cones centered around the normal
	// we will choose two orthonormal vectors (wrt normal) in the plane defined by this normal and pos	
	// get and arbitrarty perpendicaular vector towards normal (N dot A = 0)
	// [(-z-y) / x, 1, 1] is one of those vectors (unless normal is X axis)
	vec3 ortho1 = normalize(vec3(-(worldNorm.z + worldNorm.y) / worldNorm.x, 1.0f, 1.0f));
	if(worldNorm.x == 1.0f) ortho1 = vec3(0.0f, 1.0f, 0.0f);
	vec3 ortho2 = normalize(cross(worldNorm, ortho1));


	// Determine your cone's direction
	vec2 coneId = CONE_ID_MAP[globalId.x % CONE_COUNT];
	coneId = (coneId * 2.0f - 1.0f);
	coneId *= tan(coneAngle * 0.5f);
	vec3 coneDir = worldNorm + ortho1 * coneId.x + ortho2 * coneId.y;
	coneDir = normalize(coneDir);
	float coneDiameterRatio = tan(coneAngle * 0.5f) * 2.0f;

	// Start sampling towards that direction
	float totalConeOcclusion = 0.0f;
	float currentDistance = 0.0f;
	while(currentDistance < maxDistance)
	{
		// Calculate cone sphere diameter at the point
		vec3 coneRelativeLoc = coneDir * currentDistance;
		float diameter = coneDiameterRatio * currentDistance;

		// Select SVO Depth Relative to the current cone radius
		uint nodeDepth = SpanToDepth(max(1, int(ceil(diameter / worldPosSpan.w))));
		
		//DEBUG
		//nodeDepth = dimDepth.y;


		// Omit if %100 occuluded in closer ranges
		// Since its not always depth pos aligned with voxel pos
		float nodeOcclusion = SampleSVOOcclusion(worldPos + coneRelativeLoc, nodeDepth);
		float gripSpanSize = worldPosSpan.w * (0x1 << (offsetCascade.x - 1));
		bool isOmitDistance = currentDistance < (SQRT3 * gripSpanSize) && 
							  nodeOcclusion > 0.5f;
		nodeOcclusion = isOmitDistance ? 0.0f : nodeOcclusion;

		// March Distance
		float depthMultiplier =  0x1 << (dimDepth.y - nodeDepth);
		float marchDist = max(worldPosSpan.w, diameter) * sampleDistanceRatio;

		// Correction Term to prevent intersecting samples error
		nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, marchDist / (depthMultiplier * worldPosSpan.w));
		
		// Occlusion falloff (linear)
		nodeOcclusion *= (1.0f / (1.0f + currentDistance));//pow(currentDistance, 1.2f))); 
		
		// Average total occlusion value
		totalConeOcclusion += (1 - totalConeOcclusion) * nodeOcclusion;

		// Traverse Further
		currentDistance += marchDist;
	}

	// Exchange Data Between cones (total is only on leader)
	// CosTetha multiplication
	totalConeOcclusion *= dot(worldNorm, coneDir);
	SumPixelOcclusion(totalConeOcclusion);

	//totalConeOcclusion *= 3;

	// Logic Change (image write)
	if(globalId.x % CONE_COUNT == 0)
	{
		imageStore(liTex, ivec2(pixelId), vec4(vec3(1.0f - totalConeOcclusion), 0.0f));
		//imageStore(liTex, ivec2(pixelId), vec4(coneDir, 0.0f));
	}
		
}