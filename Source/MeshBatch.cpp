#include "MeshBatch.h"
#include "Globals.h"
#include "Macros.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"

// Constructors & Destructor
MeshBatch::MeshBatch(const std::vector<VertexElement>& vertexDefintion,
					 uint32_t byteStride,
					 const std::vector<std::string>& sceneFiles)

	: batchVertex(vertexDefintion, byteStride)
	, batchDrawParams()
	, batchParams(BatchParams{})
{
	IETimer timer;
	timer.Start();

	for(const std::string& file : sceneFiles)
	{
		BatchParams fileBatchParams;
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

	// All Loaded
	// Send Data then Attach Transform Index Buffer
	batchDrawParams.LockAndLoad();
	batchVertex.AttachMTransformIndexBuffer(batchDrawParams.getGLBuffer(),
											batchDrawParams.getModelTransformIndexOffset());
	timer.Stop();

	GI_LOG("");
	GI_LOG("\tDuration : %f ms", timer.ElapsedMilliS());
	GI_LOG("\tMaterials : %zd", batchParams.materialCount);
	GI_LOG("\tMeshes : %zd", batchParams.objectCount);
	GI_LOG("\tDrawPoints : %zd", batchParams.drawCallCount);
	GI_LOG("\tPolyCount : %zd", batchParams.totalPolygons);
	GI_LOG("----------");
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
    return 1;
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
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[0], IEVector3::Xaxis) * transform;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[1], IEVector3::Yaxis) * transform;
	transform = IEMatrix4x4::Rotate(gfgTransform.rotate[2], IEVector3::Zaxis) * transform;

	rotation = transform;

	transform = IEMatrix4x4::Scale(gfgTransform.scale[0], gfgTransform.scale[1], gfgTransform.scale[2]) * transform;
	transform = IEMatrix4x4::Translate({gfgTransform.translate[0], gfgTransform.translate[1], gfgTransform.translate[2]}) * transform;
}