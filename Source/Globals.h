/**

Globals For Rendering

*/


#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#include "GPUBuffer.h"
#include <AntTweakBar.h>

#define GI_CASCADE_COUNT 3

// Vertex Element
#pragma pack(push, 1)
struct VAO
{
	float vPos[3];
	float vNormal[3];
	float vUV[2];
};
struct VAOSkel
{
	float	vPos[3];
	float	vNormal[3];
	float	vUV[2];
	uint8_t	vWeight[4];
	uint8_t	vWIndex[4];
};
#pragma pack(pop)

// VertexData
#define IN_POS 0
#define IN_NORMAL 1
#define IN_UV 2
#define IN_TRANS_INDEX 3
#define IN_WEIGHT 4
#define IN_WEIGHT_INDEX 5
#define IN_LIGHT_INDEX 1

static const VertexElement elementStatic[] =
{
	{
		IN_POS,
		GPUDataType::FLOAT,
		false,
		3,
		offsetof(struct VAO, vPos),
		sizeof(VAO)
	},
	{
		IN_NORMAL,
		GPUDataType::FLOAT,
		false,
		3,
		offsetof(struct VAO, vNormal),
		sizeof(VAO)
	},
	{
		IN_UV,
		GPUDataType::FLOAT,
		false,
		2,
		offsetof(struct VAO, vUV),
		sizeof(VAO)
	}
};

static const VertexElement elementSkeletal[] =
{
	{
		IN_POS,
		GPUDataType::FLOAT,
		false,
		3,
		offsetof(struct VAOSkel, vPos),
		sizeof(VAOSkel)
	},
	{
		IN_NORMAL,
		GPUDataType::FLOAT,
		false,
		3,
		offsetof(struct VAOSkel, vNormal),
		sizeof(VAOSkel)
	},
	{
		IN_UV,
		GPUDataType::FLOAT,
		false,
		2,
		offsetof(struct VAOSkel, vUV),
		sizeof(VAOSkel)
	},
	{
		IN_WEIGHT,
		GPUDataType::UINT8,
		true,
		4,
		offsetof(struct VAOSkel, vWeight),
		sizeof(VAOSkel)
	},
	{
		IN_WEIGHT_INDEX,
		GPUDataType::UINT8,
		false,
		4,
		offsetof(struct VAOSkel, vWIndex),
		sizeof(VAOSkel)
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

// Textures (Bind Uniforms)
#define T_IN 0
#define T_COLOR 0
#define T_NORMAL 1
#define T_EDGE 1
#define T_DEPTH 2
#define T_SHADOW 3
#define T_INTENSITY 3
#define T_SHADOW_DIR 4
#define T_DENSE_NODE 5
#define T_DENSE_MAT 6

#define I_OUT 0
#define I_COLOR_FB 2
#define I_LIGHT_INENSITY 2
#define I_VOX_READ 2
#define I_VOX_WRITE 2
#define I_DEPTH_READ 0
#define I_DEPTH_WRITE 1

#define U_RENDER_TYPE 0
#define U_SHADOW_MIP_COUNT 0
#define U_TRESHOLD 0
#define U_DIRECTION 0
#define U_MAX_DISTANCE 0
#define U_DEPTH_SIZE 0
#define U_SHADOW_MAP_WH 1
#define U_CONE_ANGLE 1
#define U_NEAR_FAR 1
#define U_FETCH_LEVEL 1
#define U_PIX_COUNT 1
#define U_SAMPLE_DISTANCE 2
#define U_LIGHT_INDEX 2
#define U_IMAGE_SIZE 3
#define U_TOTAL_VOX_DIM 3
#define U_OBJ_ID 4
#define U_TOTAL_OBJ_COUNT 4
#define U_MIN_SPAN 5
#define U_MAX_CACHE_SIZE 5
#define U_MAX_GRID_DIM 6
#define U_OBJ_TYPE 6
#define U_IS_MIP 7
#define U_LIGHT_ID 4

// Unfiorm
#define U_FTRANSFORM 0
#define U_INVFTRANSFORM 1
#define U_VOXEL_GRID_INFO 2
#define U_SVO_CONSTANTS 3
#define U_CONE_PARAMS 4

// Large Uniform
#define LU_SVO_NODE 2
#define LU_LIGHT_MATRIX 0
#define LU_VOXEL_NORM_POS 0
#define LU_SVO_MATERIAL 3
#define LU_VOXEL_RENDER 1
#define LU_LIGHT 1
#define LU_OBJECT_GRID_INFO 2
#define LU_SVO_LEVEL_OFFSET 4
#define LU_VOXEL_IDS 3
#define LU_AABB 3
#define LU_MTRANSFORM 4
#define LU_INDEX_CHECK 4
#define LU_MTRANSFORM_INDEX 5
#define LU_JOINT_TRANS 6



#endif //__GLOBALS_H__