#include "MeshBatchInstanced.h"
#include "Globals.h"
#include "Macros.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"

// Constructors & Destructor
MeshBatchInstanced::MeshBatchInstanced(const char* sceneFileName,
                                       float minVoxSpan,
                                       bool isSkeletal)
    : batchVertex((isSkeletal ? Array32<const VertexElement>{elementSkeletal, 5} :
                   Array32<const VertexElement>{elementStatic, 3}))
    , batchDrawParams()
    , batchParams(BatchParams{})
    , minSpan(minVoxSpan)
{
    IETimer timer;
    timer.Start();

    std::string fileName = sceneFileName;
    batchName = fileName.substr(0, fileName.find_last_of('.'));

    GFGLoadError e = GFGLoader::LoadGFG(batchParams, batchVertex, batchDrawParams, sceneFileName, isSkeletal);
    assert(e == GFGLoadError::OK);
    batchDrawParams.SendToGPU();
    timer.Stop();

    GI_LOG("Loading \"%s\" complete", sceneFileName);
    GI_LOG("\tDuration : %f ms", timer.ElapsedMilliS());
    GI_LOG("\tMaterials : %zd", batchParams.materialCount);
    GI_LOG("\tMeshes : %zd", batchParams.objectCount);
    GI_LOG("\tDrawPoints : %zd", batchParams.drawCallCount);
    GI_LOG("\tPolyCount : %zd", batchParams.totalPolygons);
    GI_LOG("----------");
}

// Interface
void MeshBatchInstanced::Update(double elapsedS)
{}

DrawBuffer& MeshBatchInstanced::getDrawBuffer()
{
    return batchDrawParams;
}

GPUBuffer& MeshBatchInstanced::getGPUBuffer()
{
    return batchVertex;
}

const std::string& MeshBatchInstanced::BatchName() const
{
    return batchName;
}

VoxelObjectType MeshBatchInstanced::MeshType() const
{
    return VoxelObjectType::STATIC;
}

size_t MeshBatchInstanced::ObjectCount() const
{
    return batchParams.objectCount;
}

size_t MeshBatchInstanced::PolyCount() const
{
    return batchParams.totalPolygons;
}

size_t MeshBatchInstanced::MaterialCount() const
{
    return  batchParams.materialCount;
}

size_t MeshBatchInstanced::DrawCount() const
{
    return batchParams.drawCallCount;
}

float MeshBatchInstanced::MinSpan() const
{
    return minSpan;
}

void MeshBatchInstanced::GenTransformMatrix(IEMatrix4x4& transform,
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