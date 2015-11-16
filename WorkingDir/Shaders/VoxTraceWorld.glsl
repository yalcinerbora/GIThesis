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
#define U_IMAGE_SIZE layout(location = 4)

#define I_COLOR_FB layout(rgba8, binding = 0) restrict

#define LU_SVO_NODE layout(binding = 0)
#define LU_SVO_MATERIAL layout(binding = 1)

#define U_CAMERA_PARAMS layout(binding = 0)


// Uniforms
U_IMAGE_SIZE uniform uvec2 imgSize;

LU_SVO_NODE buffer SVONode
{ 
	uint svoNode[];
};

LU_SVO_MATERIAL buffer SVOMaterial
{ 
	uint svoMaterial[];
};

U_SVO_CONSTANTS uniform SVOConstants
{
	// xyz gridWorldPosition
	// w is gridSpan
	vec4 worldPosSpan;

	// x is grid dimension
	// y is grid depth
	// w is dense dimension
	// z is dense depth
	uvec4 dimDepth;
};

U_CAMERA_PARAMS uniform CameraParams
{
	uvec4 camPos;
	uvec4 camDir;
	uvec4 camUp;
};

// Textures
uniform I_COLOR_FB image2D fbo;

// SMem
// Ray stack 3 per thread
// Worst case stack usage is 5 (7, 8, 9, 10, 11)
// Other two will be stored on registers as vec4
shared unsigned int rayStack[16 * 16 * 3];

// Funcs
ivec3 WorldToVox(in vec3 worldPos)
{
	return ivec3((worldPos - wordPosSpan.xyz) / worldPosSpan.w);
}

vec3 VoxToWorld(in ivec3 voxPos, in uint depth)
{
	uint spanMultiplier = dimDepth.y - depth;
	return vec3(voxPos) * worldPosSpan.w * spanMultiplier + wordPosSpan.xyz;
}

uint PeekStack(in uvec4 rayStackHot)
{
	uint indexedDepth = rayStackHot.x - dimDepth.z;
	if (indexedDepth < 3)
	{
		uint linearLocalId = gl_LocalInvocationID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.x;
		return rayStack[linearLocalId * 3 + indexedDepth];
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

void StackCount(in uvec4 rayStackHot)
{
	return rayStackHot.x;
}

void PopStack(in uint popCount)
{
	rayStackHot.x -= popCount;
}

void PushStack(in uint nodeId)
{
	rayStackHot.x += 1;
	uint indexedDepth = rayStackHot.x - dimDepth.z;
	if (indexedDepth < 3)
	{
		uint linearLocalId = gl_LocalInvocationID.y * gl_WorkGroupSize.y + gl_LocalInvocationID.x;
		rayStack[linearLocalId * 3 + indexedDepth] = nodeId;
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

float FindMarchLength(in uvec4 rayStackHot, in vec3 marchPos, in bool inital)
{
	ivec3 voxPos = PosToVox(marchPos);

	// Cull if out of bounds
	if(any(voxPos < 0) ||
		any(voxPos >= dimDepth.x)
	{
		return -1.0f;
	}

	unsigned int nodeIndex = PeekStack(rayStackHot);
	for(unsigned int i = dimDepth.w + StackCount(rayStackHot); i <= dimDepth.y; i++)
	{
		uint currentNode;
		if(i == dimDepth.w)
		{
			ivec3 denseVox = voxPos >> (dimDepth.y - dimDepth.w);
			currentNode = svoNode[denseVox.z * dimDepth.z * dimDepth.z +
									denseVox.y * dimDepth.z + 
									denseVox.x];
		}
		else
		{
			currentNode = svoNode[dimDepth.z * dimDepth.z * dimDepth.z + nodeIndex];
		}

		if(currentNode == 0xFFFFFFFF)
		{
			// Node Empty
			// Calculate march pos
			vec3 voxWorld = VoxToWorld(voxPos, i);
			vec3 diff = marchPos - voxWorld;

			// Convert Node position and march position to voxel space

			// 6 plane intersections (skip if perpendicular (N dot Dir == 0))
		
			// choose minimum positive distance 

			// convert to march ray
			
			// Check new positions and old positions deepest common parent 
			// and pop stack until that parent

			// return minimum positive distance
		}
		else
		{
			// Node has value 
			// continue to child
			nodeIndex = currentNode + CalculateLevelChildId(voxPos, i + 1)
			StoreStack(nodeIndex);

			// Pop required
		}
		
	}
}

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	uvec2 globalId = gl_GlobalInvocationID.xy;
	if(any(greaterThanEqual(globalId, imgSize))) return;

	// Generate Ray
	ivec2 centerOffset = ivec2(imgSize) - ivec2(globalId);
	vec3 left = cross(camUp, camDir);
	vec3 rayPos = camPos;
	vec3 rayDir = cameraDir + centerOffset.x * left + centerOffset.y up;

	// Trace Ray
	uvec4 rayStackHot = uvec3(0);
	vec3 marchPos = rayPos;
	float marchLength = FindMarchLength(rayStackHot, marchPos, true);
	while(true)
	{
		if(marchLegnth == 0)
		{
			// We hit a node colorize pixel
			vec4 color = svoMaterial[DepthNodeId(rayStackHot)];
			imageStore(fbo, ivec2(globalId), color); 
		}
		else if(marchLength < 0)
		{
			// we are out of bounds pixel has background color
			return;
		}
		// We didint hit a node march the ray
		marchPos = marchLength * rayDir;
		marchLength = FindMarchLength(rayStackHot, marchPos, false);
	}
}