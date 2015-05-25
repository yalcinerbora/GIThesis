#version 430
/*	
	**Lightpass Shader**
	
	File Name	: LightPass.vert 
	Author		: Bora Yalciner
	Description	:

		LightPass Shader
*/


// Definitions
#define IN_POS layout(location = 0)

#define OUT_INDEX layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)

#define LU_LIGHT layout(std430, binding = 1)

// Input
in IN_POS vec3 vPos;
in IN_INDEX int vIndex;

// Output
flat out OUT_INDEX int fIndex;

// Uniforms
U_FTRANSFORM uniform FrameTransform
{
	mat4 view;
	mat4 projection;
	mat4 viewRotation;
};

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

void main(void)
{
	// Translate and Scale
	// Also Rotation Needed for Area Light
	mat4 modelTransform;
	// ...


	gl_Position = projection * view * model * vec4(vPos.xyz, 1.0f);
	fIndex = vIndex;
}
