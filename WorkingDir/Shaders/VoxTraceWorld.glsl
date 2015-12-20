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

#define LU_SVO_NODE layout(std430, binding = 0) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 1) readonly
#define LU_SVO_LEVEL_OFFSET layout(std430, binding = 2) readonly

#define U_RENDER_TYPE layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)

#define FLT_MAX 3.402823466e+38F
#define EPSILON 0.00001f
#define SQRT_3	1.732051f

#define RENDER_TYPE_COLOR 0
#define RENDER_TYPE_OCCULUSION 1
#define RENDER_TYPE_NORMAL 2

// Uniforms
U_RENDER_TYPE uniform uint renderType;

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
uniform I_COLOR_FB image2D fbo;

// Functions
ivec3 LevelVoxId(in vec3 worldPoint, in uint depth)
{
	ivec3 result = ivec3(floor((worldPoint - worldPosSpan.xyz) / worldPosSpan.w));
	return result >> (dimDepth.y - depth);
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

vec3 UnpackColor(in uint colorPacked)
{
	vec3 color;
	color.x = float((colorPacked & 0x000000FF) >> 0) / 255.0f;
	color.y = float((colorPacked & 0x0000FF00) >> 8) / 255.0f;
	color.z = float((colorPacked & 0x00FF0000) >> 16) / 255.0f;
	return color;
}

vec3 UnpackNormal(in uint voxNormPosY)
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

float FindMarchLength(out uint colorPacked,
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
		return FLT_MAX;
	}

	// Start tracing (stateless start from root (dense))
	unsigned int nodeIndex = 0;
	for(unsigned int i = dimDepth.w; i <= dimDepth.y; i++)
	{
		uint currentNode;
		if(i == dimDepth.w)
		{
			ivec3 denseVox = LevelVoxId(marchPos, dimDepth.w);
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


		// Color Check
		if((i < offsetCascade.w &&
		   i > (dimDepth.y - offsetCascade.x) &&
		   currentNode == 0xFFFFFFFF) ||
		   i == offsetCascade.w)
		{
			// Mid Leaf Level
			uint loc;
			if(i > dimDepth.w)
			{
				// Sparse Fetch
				loc = offsetCascade.z + svoLevelOffset[i - dimDepth.w] +
					  nodeIndex;
			}
			else
			{
				// Dense Fetch
				uint levelOffset = uint((1.0f - pow(8.0f, i)) / 
										(1.0f - 8.0f));
				uint levelDim = dimDepth.z >> (dimDepth.w - i);
				ivec3 levelVoxId = LevelVoxId(marchPos, i);
				loc = levelOffset + levelDim * levelDim * levelVoxId.z + 
					  levelDim * levelVoxId.y + 
					  levelVoxId.x;
			}
			if(renderType == RENDER_TYPE_COLOR)
				colorPacked = svoMaterial[loc].x;
			else if(renderType == RENDER_TYPE_OCCULUSION)
			{
				if(i == dimDepth.y)
				{
					float occ = UnpackOcculusion(svoMaterial[loc].x);
					occ = ceil(occ);
					colorPacked = uint(occ * 255.0f) << 24;
				}
				else
					colorPacked = svoMaterial[loc].x;
			}
			else if(renderType == RENDER_TYPE_NORMAL)
				colorPacked = svoMaterial[loc].y;
			if (colorPacked != 0) return 0.0f;
		}

		// Node check
		if(currentNode == 0xFFFFFFFF)
		{
			// Node empty 						
			// Voxel Corners are now (0,0,0) and (span, span, span)
			// span is current level grid span (leaf span * (2^ totalLevel - currentLevel)
			float levelSpan = worldPosSpan.w * float(0x1 << (dimDepth.y - i));
		
			// Convert march position to voxel space
			vec3 voxWorld = worldPosSpan.xyz + (vec3(LevelVoxId(marchPos, i)) * levelSpan);
			vec3 relativeMarchPos = marchPos - voxWorld;
		
			// Intersection check between borders of the voxel and
			// return minimum positive distance
			return IntersectDistance(relativeMarchPos, dir, levelSpan);
		}
		else
		{
			// Node has value
			// Go deeper
			nodeIndex = currentNode + CalculateLevelChildId(voxPos, i + 1);
		}	
	}
	// Code Shouldnt return from here
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
	float maxMarch = worldPosSpan.w * float(0x1 << (dimDepth.y)) * SQRT_3;
	float marchLength = 0;
	for(float totalMarch = 0.0f;
		totalMarch < maxMarch;
		totalMarch += marchLength)
	{
		uint colorOut;
		marchLength = FindMarchLength(colorOut, marchPos, rayDir);

		// March Length zero, we hit a point
		if(marchLength == 0.0f)
		{
			//vec3 color = UnpackColor(colorOut);
			//vec3 color = vec3(1.0f - UnpackOcculusion(colorOut));
			vec3 color = vec3(1.0f, 1.0f, 0.0f);
			if(renderType == RENDER_TYPE_COLOR)			   
			{
				color = UnpackColor(colorOut);
			}
			else if(renderType == RENDER_TYPE_OCCULUSION)
			{
				color = vec3(1.0f - UnpackOcculusion(colorOut));
			}
			else if(renderType == RENDER_TYPE_NORMAL)
			{
				color = UnpackNormal(colorOut);
			}			
			imageStore(fbo, ivec2(globalId), vec4(color, 0.0f)); 
			return;
		}
		else
		{
			// March Ray and Continue
			totalMarch += marchLength;
			marchPos += marchLength * rayDir;
		}
	}
	imageStore(fbo, ivec2(globalId), vec4(1.0f, 0.0f, 1.0f, 0.0f)); 
}