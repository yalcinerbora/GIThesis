#include "MeshBatchMulti.h"
#include "BatchFunctors.h"

MeshBatchMultiSkel::MeshBatchMultiSkel(const char* sceneFileName,
                                       float minVoxSpan,
                                       float xStart, float zStart,
                                       float distance,
                                       float width,
                                       int repeatCount)
    : MeshBatchSkeletal(sceneFileName, minVoxSpan, repeatCount)
    , repeatCount(repeatCount)
{
    // Edit model matrices of the object    
    std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();
    uint32_t repeatingObjCount = static_cast<uint32_t>(mtBuff.size() / repeatCount);
    for(int i = 0; i < mtBuff.size(); i++)
    {
        if(i % repeatingObjCount == 0) continue;

        float totalDistance = (i / repeatingObjCount) * distance;
        IEVector3 vector(xStart + std::fmod(totalDistance, width),
                         -10.0f,
                         zStart + static_cast<int>(totalDistance / width) * distance);
        IEMatrix4x4 trans = IEMatrix4x4::Translate(vector);
        mtBuff[i].model = trans * mtBuff[i].model;
    }
    batchDrawParams.getModelTransformBuffer().SendData();
}

int MeshBatchMultiSkel::RepeatCount() const
{
    return repeatCount;
}

MeshBatchMulti::MeshBatchMulti(const char* sceneFileName,
                                   float minVoxSpan,
                                   float xStart, float zStart,
                                   float distance,
                                   float width,
                                   int repeatCount)
    : MeshBatch(sceneFileName, minVoxSpan, false, repeatCount)
    , repeatCount(repeatCount)
{
    // Edit model matrices of the object    
    std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();
    uint32_t repeatingObjCount = static_cast<uint32_t>(mtBuff.size() / repeatCount);
    for(int i = 0; i < mtBuff.size(); i++)
    {
        if(i % repeatingObjCount == 0) continue;

        float totalDistance = (i / repeatingObjCount) * distance;
        IEVector3 vector(xStart + std::fmod(totalDistance, width),
                         -10.0f,
                         zStart + static_cast<int>(totalDistance / width) * distance);
        IEMatrix4x4 trans = IEMatrix4x4::Translate(vector);
        mtBuff[i].model = trans * mtBuff[i].model;
    }
    batchDrawParams.getModelTransformBuffer().SendData();
}

int MeshBatchMulti::RepeatCount() const
{
    return repeatCount;
}