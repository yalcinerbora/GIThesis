#version 430
/*	
	**Voxel Ambient Occulusion Compute Shader**
	
	File Name	: VoxTraceAO.vert
	Author		: Bora Yalciner
	Description	:

		Ambient Occulusion approximation using SVO
*/

#define I_COLOR_FB layout(rgba8, binding = 2) restrict writeonly

#define LU_SVO_NODE layout(std430, binding = 0) readonly
#define LU_SVO_MATERIAL layout(std430, binding = 1) readonly
#define LU_SVO_LEVEL_OFFSET layout(std430, binding = 2) readonly

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)
#define U_SVO_CONSTANTS layout(std140, binding = 3)

#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)

// Static cone count for faster implementation (prob i'll switch shaders instead of dynamically writing it)
#define CONE_COUNT 4
#define CONE_ANGLE (3.14f * 0.5f) // 90 degree
#define CONE_BEND_ANGLE (3.14f * 0.25) // 45 degree
#define TRACE_RATIO 1

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
uniform I_COLOR_FB image2D liTex;

uniform T_NORMAL usampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;

// Shared Mem
shared vec3 reduceBuff [8][8 * 2]; 

// Functions
vec3 DepthToWorld(uvec2 fragmentCood)
{
	vec2 gBuffUV = (fragmentCood - viewport.xy - vec2(0.5f)) / viewport.zw;

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

vec3 UnpackNormal(in uvec2 norm)
{
	vec3 result;
	result.x = ((float(norm.x) / 0xFFFF) - 0.5f) * 2.0f;
	result.y = ((float(norm.y & 0x7FFF) / 0x7FFF) - 0.5f) * 2.0f;
	result.z = sqrt(abs(1.0f - dot(result.xy, result.xy)));
	result.z *= sign(int(norm.y << 16));
	return result;
}

vec3 InterpolatePos(in vec3 worldPos)
{
	if(TRACE_RATIO == 1) return worldPos;
	else
	{
		// TODO:
		return worldPos;
	}

	//uvec2 localId = gl_LocalInvocationID.xy;
	//uvec2 pixelId = localId / TRACE_RATIO;
	//uvec2 pixelLocalId = localId % TRACE_RATIO;
	//uint pixelLocalLinearID = pixelLocalId.y * TRACE_RATIO + pixelLocalId.x;
	//if(pixelLocalId.x == 0) 
	//	reduceBuffer[pixelId.y][pixelId.x + pixelLocalId.y] = worldPos; 
	////memoryBarrierShared();

	//for(unsigned int i = TRACE_RATIO * TRACE_RATIO / 2; i > 0; i >>= 1)
	//{
	//	if(pixelLocalLinearID < i)
	//		worldPos = mix(worldPos, 
	//					   reduceBuffer[pixelId.y][pixelLocalLinearID + i], 
	//					   0.5f);
	//	//memoryBarrierShared();
	//}

	//if(pixelLocalLinearID
	
	////memoryBarrierShared();// This shouldnt be necessary since transaction is at warp level
	//// First Reduction
	//if(globalId.x % TRACE_RATIO == 1) 
	//	worldPos = mix(worldPos, reduceBuffer[localId.y][localId.x + (localId.y % TRACE_RATIO)], 0.5f);

	//if(globalId.x % TRACE_RATIO == 1 &&
	//   globalId.y % TRACE_RATIO == 0 &&) 
	//   reduceBuffer[localId.y][localId.x] = worldPos; 
	////memoryBarrierShared();

	//if(globalId.x % TRACE_RATIO == 1 &&
	//   globalId.y % TRACE_RATIO == 1 &&)
	//   reduceBuffer[localId.y][localId.x] = worldPos = mix(worldPos, reduceBuffer[localId.y][localId.x], 0.5f);
	////memoryBarrierShared();
}

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per cone per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	uvec2 pixelId = globalId / TRACE_RATIO;
	if(any(greaterThanEqual(pixelId, imageSize(liTex).xy)) return;

	// Fetch GBuffer (Interpolate Positions)
	vec3 worldPos = DepthToWorld(pixelId * TRACE_RATIO);
	worldPos = InterpolatePos(worldPos);

	// Each Thread Has locally same location now generate cones
	// We will cast 4 Cones centered around the normal
	// we will choose two orthonormal vectors (with normal) in the plane defined by this normal and pos	


	
	// Size conversion
	// Each cone fetches different value
	// Swap stuff between cones
	// Interpolate location and normal (or only location)
	
	// Determine your cone's direction
	
	// Sample Cone for each level (in the direction of trace)
	// Summate occlusion and write
	
	// Trace Directions
}