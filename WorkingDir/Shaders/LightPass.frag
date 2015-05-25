#version 430
/*	
	**Lightpass Shader**
	
	File Name	: LightPass.frag 
	Author		: Bora Yalciner
	Description	:

		LightPass Shader
*/

// Definitions
#define IN_INDEX layout(location = 0)

#define OUT_COLOR layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)
#define U_INVFTRANSFORM layout(std140, binding = 1)

#define LU_LIGHT layout(std430, binding = 1)

#define T_COLOR layout(binding = 0)
#define T_NORMAL layout(binding = 1)
#define T_DEPTH layout(binding = 2)
#define T_SHADOW layout(binding = 3)

// Input
flat in IN_INDEX int fIndex;

// Output
out OUT_COLOR vec4 fboColor;

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
	mat4 viewRotation;
};

U_INVFTRANSFORM uniform InverseFrameTransform
{
	mat4 invView;
	mat4 invProjection;
	mat4 invViewRotation;
}

LU_LIGHT buffer LightParams
{
	// If Position.w == 0, Its point light
	//		makes direction obselete
	// If Position.w == 1, Its directional light
	//		makes position.xyz obselete
	//		direction.w is obselete
	//		color.a is obselete
	// If Position.w == 2, Its area light
	//
	struct
	{
		vec4 position;			// position.w is the light type
		vec4 direction;			// direction.w holds shadow map index
		vec4 color;				// color.a is effecting radius
	} lightParams[];
};

// Textures
uniform T_COLOR sampler2D gBuffColor;
uniform T_NORMAL sampler2D gBuffNormal;
uniform T_DEPTH sampler2D gBuffDepth;
uniform T_SHADOW sampler2DArray shadowMaps;

void vec3 DepthToWorld()
{
	// Converts Depthbuffer Value to World Coords
	// First Depthbuffer to Screen Space
	vec3 ssPos;
	// ... 
	// ... 

	// From Screen Space to World Space
	return (invView * invProjection * vec4(ssPos)).xyz;
}

void vec3 UnpackNormal(uvec2 norm)
{

}

void vec3 PhongBDRF(in vec3 worldPos)
{
	// Phong BDRF Calculation
	// Outputs intensity multiplier for each channel (rgb)
	// Diffuse is Lambert

	// We store normals in world space in GBuffer
}

void main(void)
{
	// Do Light Calculation
	vec3 ligtIntensity = PhongBDRF(DepthToWorld());

	// Check the Light Accesssibility using shadow map
	if(....)
	{
		// Additive Blending will take care of the rest
		fboColor = vec4(lightIntensity, 1.0f);
	}
}