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

// Uniforms
uniform U_DIRECTION uint direction;

uniform vec2 DIRECTION_VECTOR[2] = 
{
	vec2(0.0f, 1.0f),
	vec2(1.0f, 0.0f)
};

uniform float WEIGHTS[9] = 
{
	1.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f
};

uniform float OFFSETS[9] = 
{
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f,
	0.0f
};

uniform I_OUT image2D imgOut;
uniform T_IN sampler2D tIn;
uniform T_EDGE sampler2D tEdge;

bool CheckEdge(vec2 uv)
{
	return dot(texture(tEdge, uv).xy, DIRECTION_VECTOR[direction]) == 1.0f;
}

layout (local_size_x = BLOCK_SIZE_X, local_size_y = BLOCK_SIZE_Y, local_size_z = 1) in;
void main(void)
{
	// Thread Logic is per pixel
	uvec2 globalId = gl_GlobalInvocationID.xy;
	vec2 uv = vec2(globalId) / vec2(imageSize(imgOut).xy);

	// Skip if Out of bounds
	if(any(greaterThanEqual(globalId, imageSize(imgOut).xy))) return;

	// Vertical or Horizontal Pass
	bvec2 foundEdge = bvec2(false);
	vec3 fragOut = texture(tIn, uv).xyz * WEIGHTS[0];
    for(int i = 1; i < 9; i++) 
	{
		vec2 uvOffset = vec2(OFFSETS[i]) * DIRECTION_VECTOR[direction] / vec2(imageSize(imgOut).xy);
		if(!foundEdge.x)
		{
			foundEdge.x = CheckEdge(uv + uvOffset);
			if(!foundEdge.x) fragOut += texture(tIn, uv + uvOffset).xyz * WEIGHTS[i];
		}

		if(!foundEdge.y)
		{
			foundEdge.y = CheckEdge(uv - uvOffset);
			if(!foundEdge.y) fragOut += texture(tIn, uv - uvOffset).xyz * WEIGHTS[i];
		}
    }
	
	imageStore(imgOut, ivec2(globalId), vec4(fragOut, 0.0f));	
}
