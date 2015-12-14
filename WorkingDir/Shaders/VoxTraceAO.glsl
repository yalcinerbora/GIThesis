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
#define CONE_COUNT_AXIS 2	// Cone count for each axis

// Ratio Difference Between LI Buffer and GBuffer
// ATM only works for 1
#define TRACE_RATIO 1

#define FLT_MAX 3.402823466e+38F
#define EPSILON 0.00001f
#define PI_OVR_2 (3.1416f * 0.5f)

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

// Shared Mem
shared float reduceBuffer[BLOCK_SIZE_X][BLOCK_SIZE_Y * (CONE_COUNT / 2)]; 

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

float IntersectDistance(in vec3 relativePos, 
						in vec3 dir, 
						in float gridDim)
{
	// 6 Plane intersection on cube normalized coordinates
	// Since planes axis aligned writing code is optimized 
	// (instead of dot products)

	// P is normCoord (ray position)
	// D is dir (ray direction)
	// N is plane normal (since axis aligned (1, 0, 0), (0, 1, 0), (0, 0, 1)
	// d is gridDim (plane distance from origin) (for "far" planes)

	// d - (P dot N) (P dot N returns Px Py Pz for each plane)
	vec3 tClose = vec3(0.0f) - relativePos;	
	vec3 tFar = vec3(gridDim) - relativePos;
	
	// Negate zeroes from direction
	// (D dot N) returns Dx Dy Dz for each plane
	// IF perpendicaular make it intersect super far
	bvec3 dirMask = greaterThan(abs(dir), vec3(EPSILON));
	dir.x = (dirMask.x) ? dir.x : EPSILON;
	dir.y = (dirMask.y) ? dir.y : EPSILON;
	dir.z = (dirMask.z) ? dir.z : EPSILON;

	// acutal T value
	// d - (P dot N) / (N dot D)
	vec3 dirInv = vec3(1.0f) / dir;
	tClose *= dirInv;
	tFar *= dirInv;

	// Negate Negative
	// Write FLT_MAX if its <= EPSILON
	bvec3 tCloseMask = greaterThan(tClose, vec3(EPSILON));
	bvec3 tFarMask = greaterThan(tFar, vec3(EPSILON));
	tClose.x = (tCloseMask.x) ? tClose.x : FLT_MAX;
	tClose.y = (tCloseMask.y) ? tClose.y : FLT_MAX;
	tClose.z = (tCloseMask.z) ? tClose.z : FLT_MAX;
	tFar.x = (tFarMask.x) ? tFar.x : FLT_MAX;
	tFar.y = (tFarMask.y) ? tFar.y : FLT_MAX;
	tFar.z = (tFarMask.z) ? tFar.z : FLT_MAX;

	// Reduction
	float minClose = min(min(tClose.x, tClose.y), tClose.z);
	float minFar = min(min(tFar.x, tFar.y), tFar.z);
	return min(minClose, minFar) + 0.1f;
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

float SampleSVOOcclusion(in vec3 worldPos, in uint depth)
{
	// Start tracing (stateless start from root (dense))
	ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);

	// Cull if out of bounds
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
	{
		return 0;
	}

	// Color Check
	if(depth <= dimDepth.w)
	{
		// Dense Fetch
		uint levelOffset = uint((1.0f - pow(8.0f, depth)) / 
								(1.0f - 8.0f));
		uint levelDim = dimDepth.z >> (dimDepth.w - depth);
		ivec3 levelVoxId = LevelVoxId(worldPos, depth);
		uint loc = levelOffset + levelDim * levelDim * levelVoxId.z + 
				   levelDim * levelVoxId.y + 
				   levelVoxId.x;
		return UnpackOcculusion(svoMaterial[loc].x);
	}
	else
	{
		unsigned int nodeIndex = 0;
		for(unsigned int i = dimDepth.w; i <= depth; i++)
		{
			uint currentNode;
			if(i == dimDepth.w)
			{
				ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
				currentNode = svoNode[denseVox.z * dimDepth.z * dimDepth.z +
									  denseVox.y * dimDepth.z + 
									  denseVox.x];
			}
			else
			{
				currentNode = svoNode[offsetCascade.y +
									  svoLevelOffset[i - dimDepth.w] +
									  nodeIndex];
			}

			// Node check (Empty node also not leaf)
			// Means object does not
			if(currentNode == 0xFFFFFFFF)
			{
				return 0;
			}
			else if(i == depth)
			{
				uint loc = offsetCascade.z + svoLevelOffset[i - dimDepth.w] +
						   nodeIndex;
				return UnpackOcculusion(svoMaterial[loc].x);
			}
			else
			{
				// Node has value
				// Go deeper
				nodeIndex = currentNode + CalculateLevelChildId(voxPos, i + 1);
			}
		}
	}
	return 0;
}

void SumPixelOcclusion(inout float totalConeOcclusion)
{
	uvec2 globald = gl_GlobalInvocationID.xy;
	uvec2 localId = gl_LocalInvocationID.xy;
	uvec2 sMemId = localId / CONE_COUNT_AXIS;
	
	uvec2 pixelConeId = localId % CONE_COUNT_AXIS;

	// left ones share their data
	if(pixelConeId.x == 0) reduceBuffer[sMemId.y][sMemId.x + pixelConeId.y] = totalConeOcclusion;
	memoryBarrierShared();

	// right ones reduce
	if(pixelConeId.x == 1)
	{
		// Lerp it at the middle (weighted avg)
		float other = reduceBuffer[sMemId.y][sMemId.x + pixelConeId.y];
		other = mix(totalConeOcclusion, other, 0.5f);

		if(all(notEqual(pixelConeId, uvec2(1u))))
			reduceBuffer[sMemId.y][sMemId.x] = other;
	}
	memoryBarrierShared();

	if(all(equal(pixelConeId, uvec2(1u))))
	{
		// Leader reduce and writes
		float other = reduceBuffer[sMemId.y][sMemId.x];
		totalConeOcclusion = mix(totalConeOcclusion, other, 0.5f);
	}
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId / CONE_COUNT_AXIS;
	if(any(greaterThanEqual(pixelId, imageSize(liTex).xy))) return;

	// Fetch GBuffer and Interpolate Positions (if size is smaller than current gbuffer)
	vec2 gBuffUV = vec2(pixelId + vec2(0.5f) - viewport.xy) / viewport.zw;
	vec3 worldPos = DepthToWorld(gBuffUV);
	vec3 worldNorm = UnpackNormalGBuff(texture(gBuffNormal, gBuffUV).xy);
	worldPos = InterpolatePos(worldPos); 
	worldNorm = InterpolateNormal(worldNorm);

	// Each Thread Has locally same location now generate cones
	// We will cast 4 Cones centered around the normal
	// we will choose two orthonormal vectors (wrt normal) in the plane defined by this normal and pos	
	// get and arbitrarty perpendicaular vector towards normal (N dot A = 0)
	vec3 ortho1 = normalize(worldNorm.xzy * vec3(0, -1.0f, 1.0f));
	vec3 ortho2 = cross(normalize(worldNorm), ortho1);

	//// Determine your cone's direction
	//uvec2 coneId = gl_GlobalInvocationID.xy % CONE_COUNT_AXIS;	// [0 or 1]
	//coneId = coneId * 2 - 1;									// [-1 or 1]
	//vec3 coneDir = worldNorm + ortho1;// * coneId.x + ortho2 * coneId.y;
	//coneDir = normalize(coneDir);
	
	vec3 coneDir = normalize(worldNorm + vec3(0.1f, 0.0f, 0.0f));
	float coneDiameterRatio = tan(coneAngle * 0.5f) * 2.0f;

	// Start sampling towards that direction
	float totalConeOcclusion = 0.0f;
	float currentDistance = 0.0f;
	while(currentDistance < maxDistance)
	{
		// Calculate cone sphere diameter at the point
		vec3 coneRelativeLoc = coneDir * currentDistance;
		float diameter = coneDiameterRatio * currentDistance;

		// Sample Location Occlusion
		uint nodeDepth = SpanToDepth(max(1, int(ceil(diameter / worldPosSpan.w))));
		float nodeOcclusion = SampleSVOOcclusion(worldPos + coneRelativeLoc, nodeDepth);

		// Omit if %100 occuluded in closer ranges
		// Since its not always depth pos aligned with voxel pos
		if(currentDistance < (worldPosSpan.w * (0x1 << (offsetCascade.x - 1))))
		nodeOcclusion = 0.0f;

		// March Distance
		float depthMultiplier =  0x1 << (dimDepth.y - nodeDepth);
		float levelSpan = worldPosSpan.w * depthMultiplier;
		vec3 voxWorld = worldPosSpan.xyz + (vec3(LevelVoxId(worldPos + coneRelativeLoc, nodeDepth)) * levelSpan);
		vec3 relativeMarchPos = worldPos + coneRelativeLoc - voxWorld;
		float marchDist = IntersectDistance(relativeMarchPos, coneDir, levelSpan);

		// Correction Term to prevent intersecting samples error
		//float multipliedSample = max(marchDist * sampleDistanceRatio, worldPosSpan.w);
		//nodeOcclusion = 1.0f - pow(1.0f - nodeOcclusion, multipliedSample / (depthMultiplier * worldPosSpan.w));
		
		// Occlusion falloff (linear)
		nodeOcclusion *= (1.0f / (1.0f + currentDistance)); 
		
		// Average total occlusion value
		totalConeOcclusion += (1 - totalConeOcclusion) * nodeOcclusion;

		// Traverse Further
		currentDistance += marchDist;
	}

	// Exchange Data Between cones (total is only on leader)
//	SumPixelOcclusion(totalConeOcclusion);
	
	// Logic Change (image write)
	if(all(equal(globalId % CONE_COUNT_AXIS, uvec2(1u))))
	{
		imageStore(liTex, ivec2(pixelId), vec4(vec3(1.0f - totalConeOcclusion), 0.0f));
		//imageStore(liTex, ivec2(pixelId), vec4(coneDir, 0.0f));
	}
		
}