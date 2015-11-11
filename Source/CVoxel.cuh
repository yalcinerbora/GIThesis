/**

Voxel Sturcutres

*/

#ifndef __CVOXEL_H__
#define __CVOXEL_H__

#include <vector_types.h>

enum class CVoxelObjectType
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

// Seperated Voxel Data
typedef uint2 CVoxelNormPos;
typedef uint2 CVoxelIds;

// Further Seperated Voxel Data
typedef unsigned int CVoxelPos;
typedef unsigned int CVoxelNorm;

// Voxel Rendering Data
#pragma pack(push, 1)
struct CVoxelRender
{
	//unsigned int	voxelTransformType;
	uchar4			color;		// Color

	// Transform Related Data
	// For Skeletal mesh these shows index of the transforms and weights
	// For Morph target this shows the neigbouring vertices and their morph related index
	//uchar4		weightIndex;
	//uchar4		weight;
};

struct CVoxelRenderSkelMorph
{
	//unsigned int	voxelTransformType;
	uchar4			color;		// Color

	// Transform Related Data
	// For Skeletal mesh these shows index of the transforms and weights
	// For Morph target this shows the neigbouring vertices and their morph related index
	uchar4			weightIndex;
	uchar4			weight;
};

#pragma pack(pop)

//
inline __device__ uint3 ExpandOnlyVoxPos(const unsigned int packedVoxX)
{
	uint3 result;
	result.x = (packedVoxX & 0x000001FF);
	result.y = (packedVoxX & 0x0003FE00) >> 9;
	result.z = (packedVoxX & 0x07FC0000) >> 18;
	return result;
}

inline __device__ float3 ExpandOnlyNormal(const unsigned int packedVoxY)
{
	float3 result;
	result.x = (static_cast<float>(packedVoxY & 0x0000FFFF) / 0x0000FFFF) * 2.0f - 1.0f;
	result.y = (static_cast<float>((packedVoxY & 0x7FFF0000) >> 16) / 0x00007FFF) * 2.0f - 1.0f;
	result.z = (((packedVoxY >> 31) == 1) ? -1.0f : 1.0f) * 1.0f - sqrtf(result.x * result.x + result.y  * result.y);
	return result;
}

inline __device__ void ExpandNormalPos(uint3& voxPos,
									   float3& normal,
									   bool& isMip,
									   const CVoxelNormPos& packedVoxNormalPos)
{
	voxPos.x = (packedVoxNormalPos.x & 0x000001FF);
	voxPos.y = (packedVoxNormalPos.x & 0x0003FE00) >> 9;
	voxPos.z = (packedVoxNormalPos.x & 0x07FC0000) >> 18;
	isMip = ((packedVoxNormalPos.x & 0xF8000000) >> 27) != 0;

	normal = ExpandOnlyNormal(packedVoxNormalPos.y);
}

inline __device__ void ExpandVoxelIds(unsigned int& voxId,
									  ushort2& objectId,
									  CVoxelObjectType& objType,
									  const CVoxelIds& packedVoxIds)
{
	objectId.x = (packedVoxIds.x & 0x0000FFFF);
	objectId.y = (packedVoxIds.x & 0x3FFF0000) >> 16;

	objType = static_cast<CVoxelObjectType>((packedVoxIds.y & 0xC0000000) >> 30);

	voxId = packedVoxIds.y;
}

inline __device__ void PackVoxelIds(CVoxelIds& packedVoxId,
									const ushort2& objId,
									const CVoxelObjectType& objType,
									const unsigned int voxRenderPtr)
{
	// 3rd word holds object id (14/16 bit each)
	// 1st is batch id second is object id on that batch
	unsigned int value = 0;
	value |= static_cast<unsigned int>(objType) << 30;
	value |= static_cast<unsigned int>(objId.y) << 16;
	value |= static_cast<unsigned int>(objId.x);
	packedVoxId.x = value;

	// Last component is voxel render index on that batch precalculated voxel batch
	packedVoxId.y = voxRenderPtr;
}

inline __device__ unsigned int PackOnlyVoxPos(const uint3& voxPos,
											  const bool isMip)
{
	unsigned int packed = 0;
	unsigned int uintMip = (isMip) ? 1 : 0;
	packed |= (uintMip & 0x0000001F) << 27;
	packed |= (voxPos.z & 0x000001FF) << 18;
	packed |= (voxPos.y & 0x000001FF) << 9;
	packed |= (voxPos.x & 0x000001FF);
	return packed;
}

inline __device__ unsigned int PackOnlyVoxNorm(const float3& normal)
{
	// (x,y components packed NORM int with 16/15 bit repectively, MSB is sign of z
	unsigned int value = 0;
	value |= signbit(normal.z) << 31;
	value |= static_cast<unsigned int>(normal.y * 0x00007FFF) << 16;
	value |= static_cast<unsigned int>(normal.x * 0x0000FFFF);
	return value;
}

inline __device__ void PackVoxelNormPos(CVoxelNormPos& packedVoxNormPos,
										const uint3& voxPos,
										const float3& normal,
										const bool isMip)
{
	// First word holds span ratio and voxel position (relative to AABB or Grid)
	packedVoxNormPos.x = PackOnlyVoxPos(voxPos, isMip);

	// Second word holds normal 
	packedVoxNormPos.y = PackOnlyVoxNorm(normal);
}
#endif //__CVOXEL_H__