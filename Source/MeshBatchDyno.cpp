#include "MeshBatchDyno.h"
#include "BatchFunctors.h"

MeshBatchDyno::MeshBatchDyno(const char* sceneFileName,
                 float minVoxSpan)
    : MeshBatch(sceneFileName, minVoxSpan, false)
{}

void MeshBatchDyno::Update(double elapsedS)
{
    // Static Indexing Etc Yolo
    // This ordering may change if maya gfg exporter decides to traverse DF differently
    // but w/e
    static constexpr uint32_t boxStart = 193;
    static constexpr uint32_t boxEnd = 257;

    static constexpr uint32_t torusStart = 0;
    static constexpr uint32_t torusEnd = 192;

    std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();

    BatchFunctors::ApplyRotation rotationFunctor(mtBuff);
    BatchFunctors::ApplyTranslation translationFunctor(mtBuff);

    // Rotation
    // Torus Rotation (Degrees per second)
    static constexpr float torusSmallSpeed = 90.5f;
    static constexpr float torusMidSpeed = 50.33f;
    static constexpr float torusLargeSpeed = 33.25f;

    static constexpr float cubeSpeedRGB = 130.123f;

    for(int i = 0; i < 64; i++)
    {
        rotationFunctor(torusStart + i * 3 + 0, torusSmallSpeed * elapsedS, IEVector3::Xaxis);
        rotationFunctor(torusStart + i * 3 + 1, torusMidSpeed * elapsedS, IEVector3::Zaxis);
        rotationFunctor(torusStart + i * 3 + 2, torusLargeSpeed * elapsedS, IEVector3::Zaxis);

        rotationFunctor(boxStart + i, cubeSpeedRGB * elapsedS, IEVector3::Xaxis);
        rotationFunctor(boxStart + i, cubeSpeedRGB * elapsedS, IEVector3::Yaxis);
    }

    batchDrawParams.getModelTransformBuffer().SendData();
}

VoxelObjectType MeshBatchDyno::MeshType() const
{
    return VoxelObjectType::DYNAMIC;
}