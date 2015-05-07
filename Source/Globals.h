/**

Globals For Rendering

*/


#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "GPUBuffer.h"

// Vertex Element
#pragma pack(push, 1)
struct VAO
{
	float vPos[3];
	float vNormal[3];
	float vUV[2];
};
#pragma pack(pop)

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
#define U_FTRANSFORM 0
#define U_MTRANSFORM 1
#define U_OBJECT 2
#define U_TOTAL_OBJ_COUNT 3

// Large Uniform
#define LU_VOXEL 0
#define LU_VOXEL_RENDER 1
#define LU_OBJECT_GRID_INFO 2
#define LU_OBJECT 3

#endif //__GLOBALS_H__