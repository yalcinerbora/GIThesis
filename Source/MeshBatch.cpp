#include "MeshBatch.h"
#include "Globals.h"
#include "Macros.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"

// Constructors & Destructor
MeshBatch::MeshBatch(const char* sceneFileName,
					 float minVoxSpan,
					 const Array32<size_t> maxVoxelCounts,
					 bool isSkeletal)
	: batchVertex((isSkeletal ? Array32<const VertexElement>{elementSkeletal, 5} : 
								Array32<const VertexElement>{elementStatic, 3}))
	, batchDrawParams()
	, batchParams(BatchParams{})
	, minSpan(minVoxSpan)
	, maxVoxelCount(maxVoxelCounts.arr, maxVoxelCounts.arr + maxVoxelCounts.length)
{
	IETimer timer;
	timer.Start();

	GFGLoadError e = GFGLoader::LoadGFG(batchParams, batchVertex, batchDrawParams, sceneFileName, isSkeletal);
	assert(e == GFGLoadError::OK);
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
const char* MeshBatch::sponzaFileName = "sponza.gfg";
const char*	MeshBatch::cornellboxFileName = "cornell.gfg";
const char* MeshBatch::sibernikFileName = "sibernik.gfg";

size_t MeshBatch::sponzaVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1900.0f),
	static_cast<size_t>(1024 * 1700.0f),
	static_cast<size_t>(1024 * 1800.0f)
};
size_t MeshBatch::cornellVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1024 * 1.5f),
	static_cast<size_t>(1024 * 1024 * 2.0f),
	static_cast<size_t>(1024 * 1024 * 1.5f)
};

size_t MeshBatch::sibernikVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1024 * 4.2f),
	static_cast<size_t>(1024 * 1024 * 2.2f),
	static_cast<size_t>(1024 * 1024 * 0.7f)
};

// Interface
void MeshBatch::Update(double elapsedS)
{}

DrawBuffer& MeshBatch::getDrawBuffer()
{
	return batchDrawParams;
}

GPUBuffer& MeshBatch::getGPUBuffer()
{
	return batchVertex;
}

size_t MeshBatch::VoxelCacheMax(uint32_t level) const
{
	assert(level < maxVoxelCount.size());
	return maxVoxelCount[level];
}

VoxelObjectType MeshBatch::MeshType() const
{
	return VoxelObjectType::STATIC;
}

size_t MeshBatch::ObjectCount() const
{
	return batchParams.objectCount;
}

size_t MeshBatch::PolyCount() const
{
	return batchParams.totalPolygons;
}

size_t MeshBatch::MaterialCount() const
{
	return  batchParams.materialCount;
}

size_t MeshBatch::DrawCount() const
{
	return batchParams.drawCallCount;
}

float MeshBatch::MinSpan() const
{
	return minSpan;
}

void MeshBatch::GenTransformMatrix(IEMatrix4x4& transform,
								   IEMatrix4x4& rotation,
								   const GFGTransform& gfgTransform)
{
	transform = IEMatrix4x4::IdentityMatrix;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[0], IEVector3::Xaxis) * transform;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[1], IEVector3::Yaxis) * transform;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[2], IEVector3::Zaxis) * transform;

	rotation = transform;

	transform = IEMatrix4x4::Scale(gfgTransform.scale[0], gfgTransform.scale[1], gfgTransform.scale[2]) * transform;
	transform = IEMatrix4x4::Translate({gfgTransform.translate[0], gfgTransform.translate[1], gfgTransform.translate[2]}) * transform;
}