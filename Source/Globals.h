#pragma once
/**

Globals For Rendering

*/

#include "VertexBuffer.h"
#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEVector4.h"
#include <AntTweakBar.h>

namespace DeviceOGLParameters
{
	extern GLint		uboAlignment;
	extern GLint		ssboAlignment;

	// Helper Functions
	size_t					SSBOAlignOffset(size_t offset);
	size_t					UBOAlignOffset(size_t offset);
	size_t					AlignOffset(size_t offset, size_t alignment);
};

#pragma pack(push, 1)
struct ModelTransform
{
	IEMatrix4x4 model;
	IEMatrix4x4 modelRotation;
};

struct AABBData
{
	IEVector4 min;
	IEVector4 max;
};
// Vertex Element
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
struct FrameTransformData
{
	IEMatrix4x4 view;
	IEMatrix4x4 projection;
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

static const std::vector<VertexElement> rigidMeshVertexDefinition =
{
	{
		VertexLogic::POSITION,
		GPUDataType::FLOAT,		
		3,
		IN_POS,
		offsetof(struct VAO, vPos),
		false
	},
	{
		VertexLogic::NORMAL,
		GPUDataType::FLOAT,		
		3,
		IN_NORMAL,
		offsetof(struct VAO, vNormal),
		false
	},
	{
		VertexLogic::UV,
		GPUDataType::FLOAT,		
		2,
		IN_UV,
		offsetof(struct VAO, vUV),
		false
	}
};

static const std::vector<VertexElement> skeletalMeshVertexDefinition =
{
	{
		VertexLogic::POSITION,
		GPUDataType::FLOAT,
		3,
		IN_POS,
		offsetof(struct VAOSkel, vPos),
		false
	},
	{
		VertexLogic::NORMAL,
		GPUDataType::FLOAT,		
		3,
		IN_NORMAL,
		offsetof(struct VAOSkel, vNormal),
		false
	},
	{
		VertexLogic::UV,
		GPUDataType::FLOAT,		
		2,
		IN_UV,
		offsetof(struct VAOSkel, vUV),
		false
	},
	{
		VertexLogic::WEIGHT,
		GPUDataType::UINT8,
		4,
		IN_WEIGHT,
		offsetof(struct VAOSkel, vWeight),
		true
	},
	{
		VertexLogic::WEIGHT_INDEX,
		GPUDataType::UINT8,
		4,
		IN_WEIGHT_INDEX,
		offsetof(struct VAOSkel, vWIndex),
		false
	}
};

extern TwType twIEVector3Type;
static const TwStructMember lightMembers[] =
{
	{ "X", TW_TYPE_FLOAT, 0, " help='X' step=0.1 " },
	{ "Y", TW_TYPE_FLOAT, 4, " help='Y' step=0.1 " },
	{ "Z", TW_TYPE_FLOAT, 8, " help='Z' step=0.1 " }
};