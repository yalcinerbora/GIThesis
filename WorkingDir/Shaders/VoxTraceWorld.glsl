#version 430
/*	
	**Voxel Raytrace Compute Shader**
	
	File Name	: VoxtraceWorld.vert
	Author		: Bora Yalciner
	Description	:

		Cuda does not support depth texture copy
		we need to copy depth values of the gbuffer to depth
*/

// Definitions
#define I_COLOR_FB layout(rgba8, binding = 2) restrict writeonly

#define LU_SVO_NODE layout(std430, binding = 0) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 1) readonly

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)

#define FLT_MAX 3.402823466e+38F
#define EPSILON 0.00001f
#define SQRT_3	1.732051f

// Uniforms
LU_SVO_NODE buffer SVONode
{ 
	uint svoNode[];
};

LU_SVO_MATERIAL buffer SVOMaterial
{ 
	uvec2 svoMaterial[];
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

// SMem
// Ray stack 3 per thread
// Worst case stack usage is 5 (7, 8, 9, 10, 11)
// Other two will be stored on registers as vec4
//shared unsigned int rayStack[16][16 * 3];

// Funcs
//uint PeekStack(in uvec4 rayStackHot)
//{
//	if(rayStackHot.x == 0) return 0;

//	uint lastIndex = rayStackHot.x - 1;
//	if (lastIndex < 3)
//	{
//		return rayStack[gl_LocalInvocationID.y]
//					   [gl_LocalInvocationID.x * 3 + lastIndex];
//	}
//	else if(lastIndex == 3)
//	{
//		return rayStackHot.y;
//	}
//	else if(lastIndex == 4)
//	{
//		return rayStackHot.z;
//	}
//	else
//	{
//		return rayStackHot.w;
//	}
//}

//uint StackCount(in uvec4 rayStackHot)
//{
//	return rayStackHot.x;
//}

//void PopStack(inout uvec4 rayStackHot, in uint popCount)
//{
//	rayStackHot.x -= popCount;
//}

//void PushStack(in uvec4 rayStackHot, in uint nodeId)
//{
//	if(rayStackHot.x < 3)
//	{
//		rayStack[gl_LocalInvocationID.y]
//  			    [gl_LocalInvocationID.x * 3 + rayStackHot.x] = nodeId;
//	}
//	else if(rayStackHot.x == 3)
//	{
//		rayStackHot.y = nodeId;
//	}
//	else if(rayStackHot.x == 4)
//	{
//		rayStackHot.z = nodeId;
//	}
//	else
//	{
//		rayStackHot.w = nodeId;
//	}
//	rayStackHot.x++;
//}

ivec3 LevelVoxId(in vec3 worldPoint, in uint depth)
{
	ivec3 result = ivec3(floor((worldPoint - worldPosSpan.xyz) / worldPosSpan.w));
	return result >> (dimDepth.y - depth);
}

vec3 PixelToWorld()
{
	vec2 gBuffUV = (vec2(gl_GlobalInvocationID.xy) + vec2(0.5f) - vec2(viewport.xy)) / vec2(viewport.zw);

	// NDC (Z is near plane)
	vec3 ndc = vec3(gBuffUV, 0.0f);
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
					  in uvec4 rayStackHot, 
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

	// Start tracing (start from cached parent)
	//unsigned int nodeIndex = PeekStack(rayStackHot);
	unsigned int nodeIndex = 0;
	for(unsigned int i = dimDepth.w/* + StackCount(rayStackHot)*/; i <= dimDepth.y; i++)
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
			currentNode = svoNode[offsetCascade.y + nodeIndex];
		}


		// Color Check
		if((i < offsetCascade.w &&
		   i > (dimDepth.y - offsetCascade.x) &&
		   currentNode == 0xFFFFFFFF) ||
		   i == offsetCascade.w)
		{
			// Mid Leaf Level
			if(i > dimDepth.w)
			{
				// Sparse Fetch
				colorPacked = svoMaterial[offsetCascade.z + nodeIndex].x;
			}
			else
			{
				// Dense Fetch
				uint levelOffset = 37449;//uint((1.0f - pow(8.0f, i)) / 
										//(1.0f - 8.0f));
				uint levelDim = dimDepth.z >> (dimDepth.w - i);
				ivec3 levelVoxId = LevelVoxId(marchPos, i);
				colorPacked = svoMaterial[levelOffset + 
											levelDim * levelDim * levelVoxId.z + 
											levelDim * levelVoxId.y + 
											levelVoxId.x].x;
			}
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
		
			// Intersection check between borders of the voxel
			float dist = IntersectDistance(relativeMarchPos, dir, levelSpan);

			//// convert to march ray
			//vec3 newMarch = marchPos + dist * dir;
					
			//// Check new positions and old positions deepest common parent 
			//ivec3 newVoxPos = LevelVoxId(newMarch, dimDepth.y);
			//ivec3 diff =  voxPos ^ newVoxPos;
			//uvec3 loc = findMSB(uvec3(~diff));
			//loc = uvec3(dimDepth.y) - (loc + 1); 
			//uint minCommon = min(min(loc.x, loc.y), loc.z);

			//// and pop stack until that parent
			//PopStack(rayStackHot, max(StackCount(rayStackHot), min(0, i - minCommon)));
			
			// return minimum positive distance
			return dist;
		}
		else
		{
			// Node has value
			// Go deeper
			nodeIndex = currentNode + CalculateLevelChildId(voxPos, i + 1);

			// Push current value to stack continue traversing
			//PushStack(rayStackHot, nodeIndex);
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

	uvec4 rayStackHot = uvec4(0);
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
		marchLength = FindMarchLength(colorOut, rayStackHot, marchPos, rayDir);

		// March Length zero, we hit a point
		if(marchLength == 0.0f)
		{
			vec3 color = UnpackColor(colorOut);
			imageStore(fbo, ivec2(gl_GlobalInvocationID.xy), vec4(color, 0.0f)); 
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