#version 430
/*	
	**Voxel Raytrace Compute Shader**
	
	File Name	: VoxTraceWorld.vert
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

#define T_DENSE_NODE layout(binding = 5)
#define T_DENSE_MAT layout(binding = 6)

#define FLT_MAX 3.402823466e+38F
#define EPSILON 0.00001f
#define SQRT_3 1.73205f

#define RENDER_TYPE_COLOR 0
#define RENDER_TYPE_OCCLUSION 1
#define RENDER_TYPE_NORMAL 2
#define RENDER_TYPE_SPECULAR 3

// Uniforms
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
uniform I_COLOR_FB image2D fbo;
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

vec3 PixelToWorld()
{
	vec2 screenUV = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5f) - vec2(viewport.xy)) / vec2(viewport.zw);

	// NDC (Z is near plane)
	vec3 ndc = vec3(screenUV, 0.0f);
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

vec4 UnpackColor(in uint colorPacked)
{
	return unpackUnorm4x8(colorPacked);
}

vec4 UnpackNormal(in uint voxNormPosY)
{
	return vec4(unpackSnorm4x8(voxNormPosY).xyz,
				unpackUnorm4x8(voxNormPosY).w);
}

float IntersectDistance(in vec3 marchPos, 
						in vec3 dir,
						in uint traverseLevel)
{
	// Voxel Corners are now (0,0,0) and (span, span, span)
	// span is current level grid span (leaf span * (2^ totalLevel - currentLevel)		
	float gridDim = worldPosSpan.w * float(0x1 << (dimDepth.y - traverseLevel));
		
	// Convert march position to voxel space
	vec3 voxWorld = worldPosSpan.xyz + (vec3(LevelVoxId(marchPos, traverseLevel)) * gridDim);
	vec3 relativePos = marchPos - voxWorld;
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

float FindMarchLength(inout vec4 outData,						
					  in vec3 marchPos,
					  in vec3 dir)
{
	ivec3 voxPos = LevelVoxId(marchPos, dimDepth.y);
	// Cull if out of bounds
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
	{
		// Node is out of bounds
		// Since cam is centered towards grid
		// Out of bounds means its cannot come towards the grid
		// directly cull
		outData = vec4(1.0f) * (1.0f - outData.w);
		return FLT_MAX;
	}

	//	 Check Dense
	if(fetchLevel <= dimDepth.w && 
	   fetchLevel >= offsetCascade.w)
	{
		// Dense Fetch
		uint mipId = dimDepth.w - fetchLevel;
		uint levelDim = dimDepth.z >> mipId;
		vec3 levelUV = LevelVoxIdF(marchPos, fetchLevel) / float(levelDim);
				
		uvec2 data = textureLod(tSVOMat, levelUV, float(mipId)).xy;
		vec4 normal = UnpackNormal(data.y);
		vec4 inData = vec4(0.0f, 0.0f, 0.0f, normal.w);
		if(renderType == RENDER_TYPE_COLOR) inData.xyz = UnpackColor(data.x).xyz;
		else if(renderType == RENDER_TYPE_OCCLUSION) inData.xyz = normal.www;
		else if(renderType == RENDER_TYPE_NORMAL) inData.xyz = normal.xyz;
		else if(renderType == RENDER_TYPE_SPECULAR) inData.xyz = UnpackColor(data.x).www;
		inData.xyz *= inData.w;
		outData += inData * (1.0f - outData.w);
		if((1.0f - outData.w) <= 0.0f) return 0.0f;
		else return IntersectDistance(marchPos, dir, fetchLevel);
	}
	else
	{
		// Start tracing (stateless start from root (dense))
		// Initialize Traverse
		unsigned int nodeIndex = 0;
		ivec3 denseVox = LevelVoxId(marchPos, dimDepth.w);
	
		// Dense Node Index Fetch	
		vec3 texCoord = vec3(denseVox) / dimDepth.z;
		nodeIndex = texture(tSVODense, texCoord).x;
		if(nodeIndex == 0xFFFFFFFF) return IntersectDistance(marchPos, dir, dimDepth.w);
		nodeIndex += CalculateLevelChildId(voxPos, dimDepth.w + 1);

		uint traversedLevel;
		for(traversedLevel = dimDepth.w + 1; 
			traversedLevel < fetchLevel;
			traversedLevel++)
		{
			uint currentNode = svoNode[offsetCascade.y + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex];
			if(currentNode == 0xFFFFFFFF && traversedLevel > (dimDepth.y - offsetCascade.x)) break;
			else if(currentNode == 0xFFFFFFFF) return IntersectDistance(marchPos, dir, traversedLevel);
			nodeIndex = currentNode + CalculateLevelChildId(voxPos, traversedLevel + 1);
		}
		if(traversedLevel > (dimDepth.y - offsetCascade.x) || 
		   traversedLevel == fetchLevel)
		{
			uint loc = offsetCascade.z + svoLevelOffset[traversedLevel - dimDepth.w] + nodeIndex;
			uvec2 mat = svoMaterial[loc].xy;

			vec4 normal = UnpackNormal(mat.y);
			if(traversedLevel == dimDepth.y) normal.w = ceil(normal.w);
			//if(normal.w == 1.0f) normal.w = 0.55f;
			vec4 inData = vec4(0.0f, 0.0f, 0.0f, normal.w);
			if(renderType == RENDER_TYPE_COLOR) inData.xyz = UnpackColor(mat.x).xyz;
			else if(renderType == RENDER_TYPE_OCCLUSION) inData.xyz = normal.www;
			else if(renderType == RENDER_TYPE_NORMAL) inData.xyz = normal.xyz;
			else if(renderType == RENDER_TYPE_SPECULAR) inData.xyz = UnpackColor(mat.x).www;
			inData.xyz *= inData.w;
			outData += inData * (1.0f - outData.w);
			if((1.0f - outData.w) <= 0.0f) return 0.0f;
			else return IntersectDistance(marchPos, dir, traversedLevel);
		}
	}
	return -1.0f;
}

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	uvec2 globalId = gl_GlobalInvocationID.xy;
	if(any(greaterThanEqual(globalId, viewport.zw))) return;

	uint linearID = gl_GlobalInvocationID.y * viewport.z +
					gl_GlobalInvocationID.x;

	// Generate Ray
	vec3 rayPos = camPos.xyz;
	vec3 rayDir = normalize(PixelToWorld() - rayPos);
	vec3 marchPos = rayPos;

	// Trace until ray is out of cascade
	// Worst case march is edge of the voxel cascade
	vec4 colorOut = vec4(0.0f);
	float maxMarch = worldPosSpan.w * float(0x1 << (dimDepth.y)) * SQRT_3 * 1.05f;
	float marchLength = 0;
	for(float totalMarch = 0.0f;
		totalMarch < maxMarch;
		totalMarch += marchLength)
	{
		marchLength = FindMarchLength(colorOut, marchPos, rayDir);
		// March Length zero, we hit a point
		if(marchLength == 0.0f) break;
		// March Ray and Continue
		totalMarch += marchLength;
		marchPos += marchLength * rayDir;
	}
	if(renderType == RENDER_TYPE_OCCLUSION) colorOut = 1.0f - colorOut;
	if(renderType == RENDER_TYPE_NORMAL) colorOut = (1.0f + colorOut) * 0.5f;
	colorOut.w = 1.0f;
	imageStore(fbo, ivec2(globalId), colorOut); 
}