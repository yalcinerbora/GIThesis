#version 430
/*	
	**Voxelize Shader**
	
	File Name	: VoxelizeGeom.geom
	Author		: Bora Yalciner
	Description	:

		Voxelizes Geometry
		Geom Shader Responsible for transforming into dominant axis
*/

#define IN_UV layout(location = 0)
#define IN_NORMAL layout(location = 1)
#define IN_POS layout(location = 2)
#define IN_WEIGHT layout(location = 4)
#define IN_WEIGHT_INDEX layout(location = 5)

#define OUT_UV layout(location = 0)
#define OUT_NORMAL layout(location = 1)
#define OUT_POS layout(location = 2)
#define OUT_WEIGHT layout(location = 4)
#define OUT_WEIGHT_INDEX layout(location = 5)

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in IN_NORMAL vec3 gNormal[];
in IN_UV vec2 gUV[];
in IN_POS vec3 gPos[];
in gl_PerVertex 
{
    vec4  gl_Position;
    //float gl_PointSize;
    //float gl_ClipDistance[];
} gl_in[];

out OUT_NORMAL vec3 fNormal;
out OUT_UV vec2 fUV;
out OUT_POS vec3 fPos;

out gl_PerVertex 
{
    vec4  gl_Position;
    //float gl_PointSize;
    //float gl_ClipDistance[];
};

const mat4 xRotate = mat4(0.0f, 0.0f, 1.0f, 0.0f,
						  0.0f, 1.0f, 0.0f, 0.0f,
						  -1.0f, 0.0f, 0.0f, 0.0f,
						  0.0f, 0.0f, 0.0f, 1.0f);

const mat4 yRotate = mat4(1.0f, 0.0f, 0.0f, 0.0f,
						  0.0f, 0.0f, 1.0f, 0.0f,
						  0.0f, -1.0f, 0.0f, 0.0f, 
						  0.0f, 0.0f, 0.0f, 1.0f);

const mat4 zRotate = mat4(1.0f, 0.0f, 0.0f, 0.0f,
						  0.0f, 1.0f, 0.0f, 0.0f,
						  0.0f, 0.0f, 1.0f, 0.0f,
						  0.0f, 0.0f, 0.0f, 1.0f);

void main(void)
{
	// Determine Dominant Axis
	// No need to normalize here we will not use this in calculation
	vec3 faceNormal = gNormal[0] + gNormal[1] + gNormal[2];
	float axisMax = max(abs(faceNormal.x), max(abs(faceNormal.y), abs(faceNormal.z)));
	vec3 axis = vec3(abs(faceNormal.x) == axisMax ? 1.0f : 0.0f,
					 abs(faceNormal.y) == axisMax ? 1.0f : 0.0f,
					 abs(faceNormal.z) == axisMax ? 1.0f : 0.0f);

	// For Each Vertex
	for(unsigned int i = 0; i < gl_in.length(); i++)
	{	
		vec4 newPos = gl_in[i].gl_Position;
		if(axis.x == 1.0f)
		{
			newPos = xRotate * gl_in[i].gl_Position;
		}	
		else if(axis.y == 1.0f)
		{
			newPos = yRotate * gl_in[i].gl_Position;
		}
		else if(axis.z == 1.0f)
		{
			newPos = zRotate * gl_in[i].gl_Position;
		}
	
		gl_Position = newPos;
		fUV = gUV[i];
		fNormal = gNormal[i];
		fPos = gPos[i];

		EmitVertex();
	}
	EndPrimitive();
}