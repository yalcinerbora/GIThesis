/**

Globals For Rendering

*/


#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "GPUBuffer.h"
#include <AntTweakBar.h>

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

extern TwType twIEVector3Type;
static const TwStructMember lightMembers[] =
{
	{ "X", TW_TYPE_FLOAT, 0, " help='X' step=0.1 " },
	{ "Y", TW_TYPE_FLOAT, 4, " help='Y' step=0.1 " },
	{ "Z", TW_TYPE_FLOAT, 8, " help='Z' step=0.1 " }
};

// Generic Binding Points
// VertexData
#define IN_POS 0
#define IN_NORMAL 1
#define IN_UV 2
#define IN_TRANS_INDEX 3
#define IN_LIGHT_INDEX 1

// Textures (Bind Uniforms)
#define T_COLOR 0
#define T_NORMAL 1
#define T_DEPTH 2
#define T_SHADOW 3
#define T_INTENSITY 3
#define T_SHADOW_DIR 4

#define I_VOX_READ 2
#define I_VOX_WRITE 2

#define U_TOTAL_VOX_DIM 3
#define U_OBJ_ID 4
#define U_TOTAL_OBJ_COUNT 4
#define U_MIN_SPAN 5
#define U_MAX_CACHE_SIZE 5
#define U_OBJ_TYPE 6
#define U_SPAN_RATIO 7
#define U_LIGHT_ID 4

// Unfiorm
#define U_FTRANSFORM 0
#define U_INVFTRANSFORM 1
#define U_VOXEL_GRID_INFO 2

// Large Uniform
#define LU_VOXEL 0
#define LU_VOXEL_RENDER 1
#define LU_OBJECT_GRID_INFO 2
#define LU_AABB 3
#define LU_MTRANSFORM 4
#define LU_INDEX_CHECK 4
#define LU_LIGHT 1
#define LU_LIGHT_MATRIX 0

#endif //__GLOBALS_H__