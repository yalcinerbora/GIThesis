#include "MeshBatchCornell.h"
#include "BatchFunctors.h"

// Constructors & Destructor
MeshBatchCornell::MeshBatchCornell(const char* sceneFileName,
								   float minVoxSpan,
								   const Array32<size_t> maxVoxelCounts)
	: MeshBatchStatic(sceneFileName, minVoxSpan, maxVoxelCounts)
{}

const char* MeshBatchCornell::cornellDynamicFileName = "cornellDynamic.gfg";
size_t MeshBatchCornell::cornellDynamicVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1024 * 1.5f),
	static_cast<size_t>(1024 * 1024 * 2.0f),
	static_cast<size_t>(1024 * 1024 * 1.5f)
};

void MeshBatchCornell::Update(double elapsedS)
{
	// only one object (cube) rotate it from both directions
	static const float sphereSpeedSlow = 70.123f;
	static const float sphereSpeedFast = 113.123f;
	std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();
	BatchFunctors::ApplyRotation rotationFunctor(mtBuff);

	rotationFunctor(1, sphereSpeedSlow * elapsedS, IEVector3::Yaxis);
	rotationFunctor(2, sphereSpeedFast * elapsedS, IEVector3::Yaxis);
	batchDrawParams.getModelTransformBuffer().SendData();
}

VoxelObjectType MeshBatchCornell::MeshType() const
{
	return VoxelObjectType::DYNAMIC;
}