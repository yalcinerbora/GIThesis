/**

Globals For Rendering

*/


#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "GPUBuffer.h"

// Vertex Element
struct VAO
{
	float vPos[3];
	float vNormal[3];
	float vUV[2];
};
static const VertexElement element[] =
{
	{
		0,
		GPUDataType::FLOAT,
		3,
		offsetof(struct VAO, vPos),
		sizeof(VAO)
	},
	{
		1,
		GPUDataType::FLOAT,
		3,
		offsetof(struct VAO, vNormal),
		sizeof(VAO)
	},
	{
		2,
		GPUDataType::FLOAT,
		2,
		offsetof(struct VAO, vUV),
		sizeof(VAO)
	}
};


// Generic Binding Points
// VertexData
#define IN_POS 0
#define IN_NORMAL 1
#define IN_UV 2

// Textures
#define T_COLOR 0
#define T_NORMAL 1

// Unfiorm
#define U_FRAME_TRANSFORM 0
#define U_MODEL_TRANSFORMS 1

// Large Uniform


#endif //__GLOBALS_H__