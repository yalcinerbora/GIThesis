#include "MeshBatchStatic.h"
#include "Globals.h"
#include "Macros.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"

// Constructors & Destructor
MeshBatchStatic::MeshBatchStatic(const char* sceneFileName,
								 float minVoxSpan,
								 const Array32<size_t> maxVoxelCounts)
	: batchVertex({element, 3})
	, batchDrawParams()
	, batchParams(BatchParams{})
	, minSpan(minVoxSpan)
	, maxVoxelCount(maxVoxelCounts.arr, maxVoxelCounts.arr + maxVoxelCounts.length)
{
	IETimer timer;
	timer.Start();

	GFGLoader::LoadGFG(batchParams, batchVertex, batchDrawParams, sceneFileName);
	batchDrawParams.SendToGPU();
	timer.Stop();

	GI_LOG("Loading \"%s\" complete", sceneFileName, timer.ElapsedMilliS());
	GI_LOG("\tDuration : %f ms", timer.ElapsedMilliS());
	GI_LOG("\tMaterials : %d", batchParams.materialCount);
	GI_LOG("\tMeshes : %d", batchParams.objectCount);
	GI_LOG("\tDrawPoints : %d", batchParams.drawCallCount);
	GI_LOG("\tPolyCount : %d", batchParams.totalPolygons);
	GI_LOG("----------");
}

// Static Files
const char* MeshBatchStatic::sponzaFileName = "sponza.gfg";
const char*	MeshBatchStatic::cornellboxFileName = "cornell.gfg";
const char* MeshBatchStatic::sibernikFileName = "sibernik.gfg";

size_t MeshBatchStatic::sponzaVoxelSizes[] = 
{
	static_cast<size_t>(1024 * 1900.0f),
	static_cast<size_t>(1024 * 1700.0f),
	static_cast<size_t>(1024 * 1800.0f)
};
size_t MeshBatchStatic::cornellVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1024 * 1.5f),
	static_cast<size_t>(1024 * 1024 * 2.0f),
	static_cast<size_t>(1024 * 1024 * 1.5f)
};

size_t MeshBatchStatic::sibernikVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1024 * 4.2f),
	static_cast<size_t>(1024 * 1024 * 2.2f),
	static_cast<size_t>(1024 * 1024 * 0.7f)
};

// Interface
void MeshBatchStatic::Update(double elapsedS)
{}

DrawBuffer& MeshBatchStatic::getDrawBuffer()
{
	return batchDrawParams;
}

GPUBuffer& MeshBatchStatic::getGPUBuffer()
{
	return batchVertex;
}

size_t MeshBatchStatic::VoxelCacheMax(uint32_t level) const
{
	assert(level < maxVoxelCount.size());
	return maxVoxelCount[level];
}

VoxelObjectType MeshBatchStatic::MeshType() const
{
	return VoxelObjectType::STATIC;
}

size_t MeshBatchStatic::ObjectCount() const
{
	return batchParams.objectCount;
}

size_t MeshBatchStatic::PolyCount() const
{
	return batchParams.totalPolygons;
}

size_t MeshBatchStatic::MaterialCount() const
{
	return  batchParams.materialCount;
}

size_t MeshBatchStatic::DrawCount() const
{
	return batchParams.drawCallCount;
}

float MeshBatchStatic::MinSpan() const
{
	return minSpan;
}