#version 430
/*	
	**32x32 Gaussian Blur Compute Shader**
	
	File Name	: EdgeDetect.glsl
	Author		: Bora Yalciner
	Description	:

	Depth and Normal Aware Edge Detection
*/

#define I_OUT layout(rg8, binding = 0) restrict writeonly

#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)

#define U_TRESHOLD layout(location = 0)
#define U_NEAR_FAR layout(location = 1)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

// Uniforms
uniform U_TRESHOLD vec2 treshold;
uniform U_NEAR_FAR vec2 nearFar;

uniform I_OUT image2D edgeMap;

uniform T_DEPTH sampler2D tDepth;
uniform T_NORMAL usampler2D tNormal;

float LinearDepth(float depthLog)
{
	float n = nearFar.x;
	float f = nearFar.y;
	return (2 * n) / (f + n - depthLog * (f - n));
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

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	ivec2 uv = ivec2(globalId);

	// Skip if Out of bounds
	if(any(greaterThanEqual(globalId, imageSize(edgeMap).xy))) return;

	// Check Your 4 Neigbour
	float d = LinearDepth(texelFetch(tDepth, uv, 0).x);
	vec3 n = UnpackNormal(texelFetch(tNormal, uv, 0).xy);

	float dx1 = LinearDepth(texelFetch(tDepth, uv + ivec2(1, 0), 0).x);
	float dx2 = LinearDepth(texelFetch(tDepth, uv + ivec2(-1, 0) , 0).x);
	float dy1 = LinearDepth(texelFetch(tDepth, uv + ivec2(0, 1), 0).x);
	float dy2 = LinearDepth(texelFetch(tDepth, uv + ivec2(0, -1) , 0).x);

	vec3 nx1 = UnpackNormal(texelFetch(tNormal, uv + ivec2(1, 0), 0).xy);
	vec3 nx2 = UnpackNormal(texelFetch(tNormal, uv + ivec2(-1, 0), 0).xy);
	vec3 ny1 = UnpackNormal(texelFetch(tNormal, uv + ivec2(0, 1), 0).xy);
	vec3 ny2 = UnpackNormal(texelFetch(tNormal, uv + ivec2(0, -1), 0).xy);

	// Start Pass
	vec2 outEdge = vec2(0.0f);

	// Check Vertial
	if(abs(d - dx1) >= treshold.x ||
	   dot(n, nx1) < treshold.y)
	   outEdge.x = 1.0f;

	if(abs(d - dx2) >= treshold.x ||
	   dot(n, nx2) < treshold.y)
	   outEdge.x = 1.0f;

	// Check Horizontal
	if(abs(d - dx1) >= treshold.x ||
	   dot(n, ny1) < treshold.y)
	   outEdge.y = 1.0f;

	if(abs(d - dy2) >= treshold.x ||
	   dot(n, ny2) < treshold.y)
	   outEdge.y = 1.0f;

	// Write
	imageStore(edgeMap, ivec2(globalId), vec4(outEdge, 0.0f, 0.0f));	
}
