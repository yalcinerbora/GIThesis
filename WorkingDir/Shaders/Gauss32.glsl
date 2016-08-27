#version 430
/*	
	**32x32 Gaussian Blur Compute Shader**
	
	File Name	: Gauss32.glsl
	Author		: Bora Yalciner
	Description	:

		Gaussian Blur Post Process
*/

#define I_OUT layout(rgba8, binding = 0) restrict writeonly

#define T_IN layout(binding = 0)
#define T_EDGE layout(binding = 1)

#define U_DIRECTION layout(location = 0)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define KERNEL_SIZE_HALF 4

// Uniforms
uniform U_DIRECTION uint direction;

uniform vec2 DIRECTION_VECTOR[2] = 
{
	vec2(0.0f, 1.0f),
	vec2(1.0f, 0.0f)
};

uniform float WEIGHTS[KERNEL_SIZE_HALF] = 
{
	0.1964825501511404,
	0.2969069646728344,
	0.09447039785044732,
	0.010381362401148057
};

uniform float OFFSETS[KERNEL_SIZE_HALF] = 
{
	0.0f,
	1.411764705882353,
	3.2941176470588234,
	5.176470588235294
};

uniform I_OUT image2D imgOut;
uniform T_IN sampler2D tIn;
uniform T_EDGE sampler2D tEdge;

bool CheckEdge(vec2 uv)
{
	return dot(texture(tEdge, uv).xy, DIRECTION_VECTOR[direction]) != 0.0f;
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	vec2 uv = vec2(globalId + 0.5f) / vec2(imageSize(imgOut).xy);

	// Skip if Out of bounds
	if(any(greaterThanEqual(globalId, imageSize(imgOut).xy))) return;

	//// Vertical or Horizontal Pass
	//vec4 fragOut = texture(tIn, uv) * WEIGHTS[0];
	//bool foundEdge;
 //   for(int i = 1; i < KERNEL_SIZE_HALF; i++) 
	//{
	//	vec2 uvOffset = OFFSETS[i] * DIRECTION_VECTOR[direction] / vec2(imageSize(imgOut).xy);
	//	//foundEdge = CheckEdge(uv + uvOffset);
	//	/*if(!foundEdge.x)*/ fragOut += texture(tIn, uv + uvOffset) * WEIGHTS[i];
		
	//	//foundEdge = CheckEdge(uv - uvOffset);
	//	/*if(!foundEdge.y)*/ fragOut += texture(tIn, uv - uvOffset) * WEIGHTS[i];
		
 //   }
	//imageStore(imgOut, ivec2(globalId), fragOut);
	
	vec2 resolution = vec2(imageSize(imgOut).xy);
	vec4 colorBase = texture2D(tIn, uv);
	vec4 result = colorBase * WEIGHTS[0];
	bvec2 edgeDetect = bvec2(false);
	for(int i = 1; i < KERNEL_SIZE_HALF; i++)
	{
		vec2 uvOffset = OFFSETS[i] * DIRECTION_VECTOR[direction] / resolution;
		
		edgeDetect.x = CheckEdge(uv + uvOffset);
		result += ((edgeDetect.x) ? colorBase : texture(tIn, uv + uvOffset)) * WEIGHTS[i];

		edgeDetect.y = CheckEdge(uv - uvOffset);
		result += ((edgeDetect.y) ? colorBase : texture(tIn, uv - uvOffset)) * WEIGHTS[i];
	}	
	imageStore(imgOut, ivec2(globalId), result);
}
