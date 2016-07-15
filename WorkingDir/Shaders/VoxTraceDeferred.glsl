#version 430
/*	
	**Voxel Deferred Sampled Compute Shader**
	
	File Name	: VoxTraceAO.vert
	Author		: Bora Yalciner
	Description	:

		Instead of tracing camera rays it directly samples deferred depth buffer to
		Sample positions from depth buffer
*/

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

// Ratio Between TraceBuffer and GBuffer
#define TRACE_RATIO 1

#define RENDER_TYPE_COLOR 0
#define RENDER_TYPE_OCCLUSION 1
#define RENDER_TYPE_NORMAL 2

#define FLT_MAX 3.402823466e+38F
#define EPSILON 0.00001f
#define PI_OVR_2 (3.1416f * 0.5f)

// Buffers
U_RENDER_TYPE uniform uint renderType;
U_FETCH_LEVEL uniform uint fetchLevel;

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
uniform I_COLOR_FB image2D traceTex;
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
	return min(minClose, minFar) + 0.01f;
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

vec3 UnpackNormalSVO(in uint voxNormPosY)
{
	return unpackSnorm4x8(voxNormPosY).xyz;
}

float UnpackOcculusion(in uint colorPacked)
{
	return unpackUnorm4x8(colorPacked).w;
	//return float((colorPacked & 0xFF000000) >> 24) / 255.0f;
}

uint SampleSVO(in vec3 worldPos)
{
	// Start tracing (stateless start from root (dense))
	ivec3 voxPos = LevelVoxId(worldPos, dimDepth.y);

	// Cull if out of bounds
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
	{
		return 0;
	}

	// Check Dense
	if(fetchLevel <= dimDepth.w &&
	   fetchLevel >= offsetCascade.w)
	{
		// Dense Fetch
		uint mipId = dimDepth.w - fetchLevel;
		uint levelDim = dimDepth.z >> mipId;
		vec3 levelUV = LevelVoxId(worldPos, fetchLevel) / float(levelDim);
				
		if(renderType == RENDER_TYPE_COLOR)
			return textureLod(tSVOMat, levelUV, float(mipId)).x;
		else if(renderType == RENDER_TYPE_OCCLUSION)
			return textureLod(tSVOMat, levelUV, float(mipId)).y;
		else if(renderType == RENDER_TYPE_NORMAL)
			return textureLod(tSVOMat, levelUV, float(mipId)).y;
	}

	// Sparse Check
	unsigned int nodeIndex = 0;
	for(unsigned int i = dimDepth.w; i <= dimDepth.y; i++)
	{
		uint currentNode;
		if(i == dimDepth.w)
		{
			ivec3 denseVox = LevelVoxId(worldPos, dimDepth.w);
			vec3 texCoord = vec3(denseVox) / dimDepth.z;
			currentNode = texture(tSVODense, texCoord).x;
		}
		else
		{
			currentNode = svoNode[offsetCascade.y +
								  svoLevelOffset[i - dimDepth.w] +
								  nodeIndex];
		}
		
		// Color Check
		if((i < fetchLevel &&
		   i > (dimDepth.y - offsetCascade.x) &&
		   currentNode == 0xFFFFFFFF) ||
		   i == fetchLevel)
		{
			// Mid Leaf Level
			uint loc = offsetCascade.z + svoLevelOffset[i - dimDepth.w] + nodeIndex;
			if(renderType == RENDER_TYPE_COLOR)
				return svoMaterial[loc].x;
			else if(renderType == RENDER_TYPE_OCCLUSION)
			{
				if(i == dimDepth.y)
				{
					float occ = UnpackOcculusion(svoMaterial[loc].y);
					occ = ceil(occ);
					return uint(occ * 255.0f) << 24;
				}
				else
					return svoMaterial[loc].y;
			}
			else if(renderType == RENDER_TYPE_NORMAL)
				return svoMaterial[loc].y;
		}

		// Node check (Empty node also not leaf)
		// Means object does not
		if(currentNode == 0xFFFFFFFF)
		{
			return 0;
		}
		else
		{
			// Node has value
			// Go deeper
			nodeIndex = currentNode + CalculateLevelChildId(voxPos, i + 1);
		}
	}
	return 0;
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

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId / TRACE_RATIO;
	if(any(greaterThanEqual(pixelId, imageSize(traceTex).xy))) return;

	// Fetch GBuffer and Interpolate Positions (if size is smaller than current gbuffer)
	vec2 gBuffUV = vec2(pixelId + vec2(0.5f) - viewport.xy) / viewport.zw;
	vec3 worldPos = DepthToWorld(gBuffUV);
	worldPos = InterpolatePos(worldPos); 

	uint data = SampleSVO(worldPos);
	vec3 color = vec3(1.0f, 0.0f, 1.0f);
	if(data != 0)
	{
		if(renderType == RENDER_TYPE_COLOR)			   
		{
			color = UnpackColor(data);
		}
		else if(renderType == RENDER_TYPE_OCCLUSION)
		{
			color = vec3(1.0f - UnpackOcculusion(data));
		}
		else if(renderType == RENDER_TYPE_NORMAL)
		{
			color = UnpackNormalSVO(data);
		}
	}
	imageStore(traceTex, ivec2(globalId), vec4(color, 0.0f)); 
}