#ifndef __VOXELSCENECACHE_H__
#define __VOXELSCENECACHE_H__

#include <cstdint>
#include "StructuredBuffer.h"
#include "VoxelCacheData.h"
#include "VoxelDebugVAO.h"

#pragma pack(push, 1)
struct ObjGridInfo
{
	float span;
	uint32_t voxCount;
};

struct VoxelGridInfoGL
{
	IEVector4		posSpan;
	uint32_t		dimension[4];
};
#pragma pack(pop)

struct VoxelObjectCache
{
	StructuredBuffer<VoxelNormPos>			voxelNormPos;
	StructuredBuffer<VoxelIds>				voxelIds;
	StructuredBuffer<VoxelColorData>		voxelRenderData;
	StructuredBuffer<VoxelWeightData>		voxelWeightData;
	StructuredBuffer<ObjGridInfo>			objInfo;	
	VoxelDebugVAO							voxelVAO;

	bool									isSkeletal;

	uint32_t								batchVoxCacheCount;
	double									batchVoxCacheSize;

	VoxelObjectCache(size_t voxelCount, size_t objCount, bool isSkeletal)
		: isSkeletal(isSkeletal)
		, voxelNormPos(voxelCount)
		, voxelIds(voxelCount)
		, objInfo(objCount)
		, voxelRenderData(voxelCount)
		, voxelWeightData((isSkeletal) ? voxelCount : 0)
		, voxelVAO(voxelNormPos, voxelIds, voxelRenderData, voxelWeightData, isSkeletal)
		, batchVoxCacheCount(static_cast<uint32_t>(voxelCount))
		, batchVoxCacheSize(VoxelCacheSizeMB(voxelCount))
	{}

	VoxelObjectCache(VoxelObjectCache&& other)
		: isSkeletal(other.isSkeletal)
		, voxelNormPos(std::move(other.voxelNormPos))
		, voxelIds(std::move(other.voxelIds))
		, objInfo(std::move(other.objInfo))
		, voxelRenderData(std::move(other.voxelRenderData))
		, voxelVAO(std::move(other.voxelVAO))
		, batchVoxCacheCount(other.batchVoxCacheCount)
		, batchVoxCacheSize(other.batchVoxCacheSize)
	{}

	VoxelObjectCache(const VoxelObjectCache&) = delete;

	static double VoxelCacheSizeMB(size_t voxelCount)
	{
		return voxelCount * (sizeof(VoxelNormPos) + /*sizeof(VoxelIds) +*/ sizeof(VoxelColorData)) /
				1024.0 / 1024.0;
	}
};

struct SceneVoxCache
{
	uint32_t						depth;
	float							span;
	std::vector<VoxelObjectCache>	cache;

	uint32_t						voxOctreeCount;
	double							voxOctreeSize;

	uint32_t						totalCacheCount;
	double							totalCacheSize;

	SceneVoxCache() = default;

	SceneVoxCache(SceneVoxCache&& other)
		: depth(other.depth)
		, span(other.span)
		, cache(std::move(other.cache))
		, voxOctreeCount(other.voxOctreeCount)
		, voxOctreeSize(other.voxOctreeSize)
		, totalCacheCount(other.totalCacheCount)
		, totalCacheSize(other.totalCacheSize)
	{}

	SceneVoxCache(const SceneVoxCache&) = delete;
};
#endif //__VOXELSCENECACHE_H__