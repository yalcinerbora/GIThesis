#include "MeshBatch.h"
#include "Globals.h"
#include "Macros.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"

MeshBatch::MeshBatch()
	: batchParams{0, 0, 0, 0}
	, repeatCount(0)
{}

MeshBatch::MeshBatch(const std::vector<VertexElement>& vertexDefintion,
					 uint32_t byteStride,
					 const std::vector<std::string>& sceneFiles,
					 uint32_t repeatCount)
	: batchVertex(vertexDefintion, byteStride)
	, batchDrawParams()
	, batchParams{}
	, repeatCount(repeatCount)
{
	if(sceneFiles.empty()) return;

	IETimer timer;
	timer.Start();

	for(const std::string& file : sceneFiles)
	{
		BatchParams fileBatchParams = {};
		auto err = GFGLoader::LoadGFG(fileBatchParams,
									  batchVertex,
									  batchDrawParams,
									  file);

		batchParams.drawCallCount += fileBatchParams.drawCallCount;
		batchParams.materialCount += fileBatchParams.materialCount;
		batchParams.objectCount += fileBatchParams.objectCount;
		batchParams.totalPolygons += fileBatchParams.totalPolygons;

		assert(err == GFGLoadError::OK);
		if(err != GFGLoadError::OK) return;
		GI_LOG("Loading \"%s\" complete", file.c_str());
	}

	if(repeatCount > 1)
	{
		batchDrawParams.RepeatDrawCalls(repeatCount - 1);
		batchParams.drawCallCount = batchDrawParams.getDrawPointCount();
	}

	// All Loaded
	// Send Data then Attach Transform Index Buffer
	batchDrawParams.LockAndLoad();
	batchVertex.LockAndLoad();
	batchVertex.AttachMTransformIndexBuffer(batchDrawParams.getGLBuffer(),
											batchDrawParams.getModelTransformIndexOffset());
	timer.Stop();

	GI_LOG("");
	GI_LOG("\tDuration   : %f ms", timer.ElapsedMilliS());
	GI_LOG("\tMaterials  : %zd", batchParams.materialCount);
	GI_LOG("\tMeshes     : %zd", batchParams.objectCount);
	GI_LOG("\tDrawPoints : %zd", batchParams.drawCallCount);
	GI_LOG("\tPolyCount  : %zd", batchParams.totalPolygons);
	GI_LOG("----------");
}

MeshBatch::MeshBatch(MeshBatch&& other)
	: batchVertex(std::move(other.batchVertex))
	, batchDrawParams(std::move(other.batchDrawParams))
	, batchParams(std::move(other.batchParams))
	, repeatCount(other.repeatCount)
{}

MeshBatch& MeshBatch::operator=(MeshBatch&& other)
{
	assert(this != &other);
	batchVertex = std::move(other.batchVertex);
	batchDrawParams = std::move(other.batchDrawParams);
	batchParams = std::move(other.batchParams);
	repeatCount = other.repeatCount;
	return *this;
}

// Interface
void MeshBatch::Update(double elapsedS)
{}

DrawBuffer& MeshBatch::getDrawBuffer()
{
	return batchDrawParams;
}

VertexBuffer& MeshBatch::getVertexBuffer()
{
	return batchVertex;
}

MeshBatchType MeshBatch::MeshType() const
{
	return MeshBatchType::RIGID;
}

int MeshBatch::RepeatCount() const
{
    return repeatCount;
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

void MeshBatch::GenTransformMatrix(IEMatrix4x4& transform,
								   IEMatrix4x4& rotation,
								   const GFGTransform& gfgTransform)
{
	transform = IEMatrix4x4::IdentityMatrix;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[0], IEVector3::XAxis) * transform;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[1], IEVector3::YAxis) * transform;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[2], IEVector3::ZAxis) * transform;

	rotation = transform;

	transform = IEMatrix4x4::Scale(gfgTransform.scale[0], gfgTransform.scale[1], gfgTransform.scale[2]) * transform;
	transform = IEMatrix4x4::Translate({gfgTransform.translate[0], gfgTransform.translate[1], gfgTransform.translate[2]}) * transform;
}