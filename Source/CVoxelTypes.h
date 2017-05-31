#pragma once
/**



*/
#include "COpenGLTypes.h"
#include "VoxelizerTypes.h"
#include <vector_types.h>
#include <cstdint>

enum class CObjectType
{
	STATIC,			// Object does not move
	DYNAMIC,		// Object does move (with transform matrices)
	SKEL_DYNAMIC,	// Object moves with weighted transformation matrices
	MORPH_DYNAMIC,	// Object moves with morph targets (each voxel has their adjacent vertex morphs weighted)
};

// Global Voxel Data
struct CVoxelGrid
{
	float3			position;	// World Position of the voxel grid
	float			span;
	uint3			dimension;	// Voxel Grid Dimentions
	unsigned int	depth;
};

// Further Seperated Voxel Data
typedef VoxelPosition CVoxelPos;
typedef VoxelNormal CVoxelNorm;
typedef uint32_t CVoxelOccupancy;
typedef uchar4 CVoxelAlbedo;
typedef MeshVoxelInfo CMeshVoxelInfo;

// Voxel Rendering Data
#pragma pack(push, 1)
struct CVoxelWeights
{
	uchar4 weight;
	uchar4 weightIndex;
};
#pragma pack(pop)

static_assert(sizeof(CVoxelAlbedo) == sizeof(VoxelAlbedo), "Voxel albedo size mismatch.");
static_assert(sizeof(CVoxelWeights) == sizeof(VoxelWeights), "Voxel weþght size mismatch.");

enum class CSegmentOccupation : unsigned char
{
	EMPTY = 0,
	OCCUPIED = 1,
	MARKED_FOR_CLEAR = 2,
};

struct CSegmentInfo
{
	uint16_t				batchId;
	uint16_t				objId;
	uint16_t				objectSegmentId;
	uint16_t				packed;	// MSB to LSB
									// 2 bit cascadeNo
									// 2 bit objectType
									// 2 bit occupation
									// 10 bit unused
};

struct CVoxelPage
{
	CVoxelPos*				dGridVoxPos;
	CVoxelNorm*				dGridVoxNorm;
	CVoxelOccupancy*		dGridVoxOccupancy;
	unsigned char*			dEmptySegmentPos;
	CSegmentInfo*			dSegmentInfo;
	unsigned int			dEmptySegmentStackSize;
};

struct CVoxelPageConst
{
	const CVoxelPos*		dGridVoxPos;
	const CVoxelNorm*		dGridVoxNorm;
	const CVoxelOccupancy*	dGridVoxOccupancy;
	const unsigned char*	dEmptySegmentPos;
	const CSegmentInfo*		dSegmentInfo;
	const unsigned int		dEmptySegmentStackSize;
};

struct BatchVoxelCache
{
	const CMeshVoxelInfo*	dMeshVoxelInfo;
	const CVoxelPos*		dVoxelPos;
	const CVoxelNorm*		dVoxelNorm;
	const CVoxelAlbedo*		dVoxelAlbedo;
	const CVoxelWeights*	dVoxelWeight;
};

// Mapped Batch Pointers
struct BatchOGLData
{	
	const CAABB*			dAABBs;
	const uint32_t*			dModelTransformIndices;
	const CModelTransform*	dModelTransforms;
	const CJointTransform*	dJointTransforms;
};