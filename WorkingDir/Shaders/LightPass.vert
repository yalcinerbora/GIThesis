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
#define IN_INDEX layout(location = 1)

#define OUT_INDEX layout(location = 0)

#define U_FTRANSFORM layout(std140, binding = 0)

#define LU_LIGHT layout(std430, binding = 1)

#define GI_LIGHT_POINT 0.0f
#define GI_LIGHT_DIRECTIONAL 1.0f
#define GI_LIGHT_AREA 2.0f

// Input
in IN_POS vec3 vPos;
in IN_INDEX uint vIndex;

// Output
out gl_PerVertex {vec4 gl_Position;};	// Mandatory
flat out OUT_INDEX uint fIndex;

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
	mat4 model;
	if(lightParams[vIndex].position.w == GI_LIGHT_AREA)
	{
		// Area Light
		// Area light has half sphere directed towards -y direction
		float scaleFactor = lightParams[vIndex].color.w;
		vec3 translate = lightParams[vIndex].position.xyz;
		model = mat4 (scaleFactor,	0.0f,			0.0f,		 0.0f,
					  0.0f,			scaleFactor,	0.0f,		 0.0f,
					  0.0f,			0.0f,			scaleFactor, 0.0f,
					  //0.0f,			0.0f,			0.0f,		 1.0f);
					  translate.x,	translate.y,	translate.z,	1.0f);

		// Add direction rotation to the matrix
		//vec3 axis = cross(vec3(0.0f, -1.0f, 0.0f), lightParams[vIndex].direction.xyz);
		//float cosAngle = dot(vec3(0.0f, -1.0f, 0.0f), lightParams[vIndex].direction.xyz);
		//float t = 1.0f - cosAngle;
		//float sinAngle = length(axis);

		//vec3 tt = t * vec3(axis.y * axis.z, axis.x * axis.z, axis.x * axis.y);
		//vec3 st = sinAngle * axis;
		//vec3 dt = vec3(cosAngle) + (axis * axis) * t;

		//vec3 sum = tt + st;
		//vec3 diff = tt - st;
		//model *= mat4 (dt.x,		diff.z,			sum.y,			0.0f,
		//			   sum.z,		dt.y,			diff.x,			0.0f,
		//			   diff.y,		sum.x,			dt.z,			0.0f,
		//			   translate.x,	translate.y,	translate.z,	1.0f);
	}
	else if(lightParams[vIndex].position.w == GI_LIGHT_POINT)
	{
		// Point Light
		// Its unit sphere so only translate the sphere to the light position
		// and scale according to the radius
		float scaleFactor = lightParams[vIndex].color.w;
		vec3 translate = lightParams[vIndex].position.xyz;
		model = mat4 (scaleFactor,	0.0f,			0.0f,		 0.0f,
					  0.0f,			scaleFactor,	0.0f,		 0.0f,
					  0.0f,			0.0f,			scaleFactor, 0.0f,
					  translate.x,	translate.y,	translate.z, 1.0f);
	}
	else
	{
		// Directional Light
		// Its post process triangle
		gl_Position = vec4(vPos.xyz, 1.0f);
		fIndex = vIndex;
		return;
	}
	gl_Position = projection * view * model * vec4(vPos.xyz, 1.0f);
	fIndex = vIndex;
}