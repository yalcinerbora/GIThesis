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
#define U_IMAGE_SIZE layout(location = 3)

#define I_COLOR_FB layout(rgba8, binding = 2) restrict writeonly

#define LU_SVO_NODE layout(binding = 0) readonly
#define LU_SVO_MATERIAL layout(binding = 1) readonly

#define U_CAMERA_PARAMS layout(binding = 2)
#define U_SVO_CONSTANTS layout(binding = 3)

#define FLT_MAX 3.402823466e+38F

// Uniforms
U_IMAGE_SIZE uniform uvec2 imgSize;

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
	uvec4 offsetCascade;
};

U_CAMERA_PARAMS uniform CameraParams
{
	vec4 camPos;
	vec4 camDir;
	vec4 camUp;

	// X is tan(fovX/2)
	// Y is tan(fovY/2)
	// Z is "near"
	// W is "far"
	vec4 halfFoxNearFar;
};

// Textures
uniform I_COLOR_FB image2D fbo;

// SMem
// Ray stack 3 per thread
// Worst case stack usage is 5 (7, 8, 9, 10, 11)
// Other two will be stored on registers as vec4
shared unsigned int rayStack[16][16 * 3];

// Funcs
ivec3 WorldToVox(in vec3 worldPos)
{
	return ivec3((worldPos - worldPosSpan.xyz) / worldPosSpan.w);
}

vec3 VoxToWorld(in ivec3 voxPos, in uint depth)
{
	uint spanMultiplier = dimDepth.y - depth;
	return vec3(voxPos) * worldPosSpan.w * spanMultiplier + worldPosSpan.xyz;
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

float IntersectDistance(in vec3 normCoord, 
						in vec3 dir, 
						in float gridDim)
{
	// 6 Plane intersection on cube normalized coordinates
	// Since planes axis aligned writing code optimized 
	// (instead of dot products
	vec3 tClose = -normCoord;
	vec3 tFar = -(normCoord + gridDim);
	tClose /= dir;
	tFar /= dir;

	// Negate Negative
	bvec3 tCloseMask = greaterThan(tClose, vec3(0.0f));
	bvec3 tFarMask = greaterThan(tFar, vec3(0.0f));

	// Write FLT_MAX if its <= 0
	tClose.x = (tCloseMask.x) ? tClose.x : FLT_MAX;
	tClose.y = (tCloseMask.y) ? tClose.y : FLT_MAX;
	tClose.z = (tCloseMask.z) ? tClose.z : FLT_MAX;
	tFar.x = (tFarMask.x) ? tFar.x : FLT_MAX;
	tFar.y = (tFarMask.y) ? tFar.y : FLT_MAX;
	tFar.z = (tFarMask.z) ? tFar.z : FLT_MAX;

	// Reduction
	float minClose = min(min(tClose.x, tClose.y), tClose.z);
	float minFar = min(min(tFar.x, tFar.y), tFar.z);

	// Boost a little bit to be sure
	return min(minClose, minFar);
}

ivec3 LevelVoxPos(in ivec3 voxPos, in uint level)
{
	return voxPos >> (dimDepth.y - level);
}

uint PeekStack(in uvec4 rayStackHot)
{
	uint indexedDepth = rayStackHot.x - dimDepth.z;
	if (indexedDepth < 3)
	{
		return rayStack[gl_LocalInvocationID.y]
					   [gl_LocalInvocationID.x * 3 + indexedDepth];
	}
	else if(indexedDepth == 3)
	{
		return rayStackHot.y;
	}
	else if(indexedDepth == 4)
	{
		return rayStackHot.z;
	}
	else
	{
		return rayStackHot.w;
	}
}

uint StackCount(in uvec4 rayStackHot)
{
	return rayStackHot.x;
}

void PopStack(inout uvec4 rayStackHot, in uint popCount)
{
	rayStackHot.x -= popCount;
}

void PushStack(in uvec4 rayStackHot, in uint nodeId)
{
	rayStackHot.x += 1;
	uint indexedDepth = rayStackHot.x - dimDepth.z;
	if(indexedDepth < 3)
	{
		rayStack[gl_LocalInvocationID.y]
  			    [gl_LocalInvocationID.x * 3 + indexedDepth] = nodeId;
	}
	else if(indexedDepth == 3)
	{
		rayStackHot.y = nodeId;
	}
	else if(indexedDepth == 4)
	{
		rayStackHot.z = nodeId;
	}
	else
	{
		rayStackHot.w = nodeId;
	}
}

float FindMarchLength(in uvec4 rayStackHot, 
					  in vec3 marchPos,
					  in vec3 dir)
{
	ivec3 voxPos = WorldToVox(marchPos);

	// Cull if out of bounds
	if(any(lessThan(voxPos, ivec3(0))) ||
	   any(greaterThanEqual(voxPos, ivec3(dimDepth.x))))
	{
		//// Node is out of bounds but does it coming towards to grid?
		//vec3 normMarchPos = marchPos - worldPosSpan.xyz;
			
		//// 6 plane intersections (skip if perpendicular (N dot Dir == 0))
		//float dist = IntersectDistance(normMarchPos, dir, 
		//							   worldPosSpan.w * float(0x1 << (dimDepth.y)));
		//return dist;

		return FLT_MAX;
	}
	return 0.0f;

	//unsigned int nodeIndex = PeekStack(rayStackHot);
	//for(unsigned int i = dimDepth.w + StackCount(rayStackHot); i <= dimDepth.y; i++)
	//{
	//	uint currentNode;
	//	if(i == dimDepth.w)
	//	{
	//		ivec3 denseVox = LevelVoxPos(voxPos, dimDepth.w);
	//		currentNode = svoNode[denseVox.z * dimDepth.z * dimDepth.z +
	//								denseVox.y * dimDepth.z + 
	//								denseVox.x];
	//	}
	//	else
	//	{
	//		currentNode = svoNode[offsetCascade.y + nodeIndex];
	//	}

	//	if(currentNode == 0xFFFFFFFF)
	//	{
	//		// Node maybe Empty
	//		// This may contain color
	//		// Check Material
	//		if((dimDepth.y - i) < offsetCascade.x)
	//		{
	//			// Its leaf cascades, check material color
	//			uint colorPacked = svoMaterial[offsetCascade.z + nodeIndex].x;
	//			if (colorPacked != 0)
	//			{
	//				// This node contains color write image and return
	//				vec3 color = UnpackColor(colorPacked);
	//				imageStore(fbo, ivec2(gl_GlobalInvocationID.xy), vec4(color, 0.0f)); 
	//				return 0.0f;
	//			}
	//		}
			
	//		// Node empty 						
	//		// Convert Node position and march position to voxel space
	//		vec3 voxWorld = VoxToWorld(voxPos, i);
	//		vec3 normMarchPos = marchPos - voxWorld;
			
	//		// 6 plane intersections (skip if perpendicular (N dot Dir == 0))
	//		float dist = IntersectDistance(normMarchPos, dir, 
	//									   worldPosSpan.w * float(0x1 << (dimDepth.y - i + 1)));

	//		// convert to march ray
	//		vec3 newMarch = marchPos + dist * dir;
					
	//		// Check new positions and old positions deepest common parent 
	//		ivec3 newVoxPos = WorldToVox(newMarch);
	//		ivec3 diff =  voxPos ^ newVoxPos;
	//		uvec3 loc = findMSB(uvec3(~diff));
	//		loc -= dimDepth.y; 
	//		uint minCommon = min(min(loc.x, loc.y), loc.z);

	//		// and pop stack until that parent
	//		PopStack(rayStackHot, (i - 1) - minCommon);
			
	//		// return minimum positive distance
	//		return dist;
	//	}
	//	else
	//	{
	//		// Node has value
	//		// Push current value to stack continue traversing
	//		PushStack(rayStackHot, nodeIndex);
	//		nodeIndex = currentNode + CalculateLevelChildId(voxPos, i + 1);
	//	}	
	//}
	// Code Shouldnt return from here
	//return 0.0f;
}

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	uvec2 globalId = gl_GlobalInvocationID.xy;
	if(any(greaterThanEqual(globalId, imgSize))) return;

	//uint linearID = gl_GlobalInvocationID.y * imgSize.x +
	//				gl_GlobalInvocationID.x;

	//vec4 color;
	//color.xyz = UnpackColor(svoMaterial[linearID].x);
	//color.w = 1.0f;

	//if(all(equal(color.xyz, vec3(41.0f / 255.0f, 34.0f / 255.0f, 109.0f / 255.0f))) ||
	//	all(equal(color.xyz, vec3(161.0f / 255.0f, 17.0f / 255.0f, 13.0f / 255.0f))) ||
	//	all(equal(color.xyz, vec3(185.0f / 255.0f, 181.0f / 255.0f, 173.0f / 255.0f))))
	//	imageStore(fbo, ivec2(globalId), vec4(0.0f)); 
	//else
	//	imageStore(fbo, ivec2(globalId), color); 

	// Generate Ray
	vec2 ndcCoords = ((vec2(globalId) + 0.5f) / vec2(imgSize) * 2.0f) - 1.0f; // [-1, 1]
	ndcCoords *= halfFoxNearFar.xy;
	vec3 left = cross(camUp.xyz, camDir.xyz);
	vec3 rayPos = camPos.xyz;
	vec3 rayDir = normalize((camDir.xyz * halfFoxNearFar.z) + 
								ndcCoords.x * (-left) + 
								ndcCoords.y * camUp.xyz);

	//imageStore(fbo, ivec2(globalId), vec4(rayDir, 0.0f)); 
	
	uvec4 rayStackHot = uvec4(0);
	vec3 marchPos = rayPos;
	float marchLength = 0;

	marchLength = FindMarchLength(rayStackHot, marchPos, rayDir);
		
	if(marchLength == FLT_MAX) return;

	imageStore(fbo, ivec2(globalId), vec4(rayDir, 0.0f)); 


	//// Trace until ray is out of view frustum
	//for(float totalMarch = 0.0f;
	//	totalMarch < halfFoxNearFar.w;
	//	totalMarch += marchLength)
	//{

	//}
}