#include "MeshBatchCube.h"
#include "BatchFunctors.h"

// Constructors & Destructor
MeshBatchCube::MeshBatchCube(const char* sceneFileName,
							 float minVoxSpan)
	: MeshBatch(sceneFileName, minVoxSpan, false)
{}

const char* MeshBatchCube::rotatingCubeFileName = "rainbowCube.gfg";
size_t MeshBatchCube::rotatingCubeVoxelSizes[] =
{
	static_cast<size_t>(1024 * 120.0f),
	static_cast<size_t>(1024 * 35.0f),
	static_cast<size_t>(1024 * 10.0f)
};

// Interface
void MeshBatchCube::Update(double elapsedS)
{
	// only one object (cube) rotate it from both directions
	static const float cubeSpeedRGB = 130.123f;
	std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();
	BatchFunctors::ApplyRotation rotationFunctor(mtBuff);
	rotationFunctor(1, cubeSpeedRGB * elapsedS, IEVector3::Xaxis);
	rotationFunctor(1, cubeSpeedRGB * elapsedS, IEVector3::Yaxis);
	batchDrawParams.getModelTransformBuffer().SendData();
}

VoxelObjectType MeshBatchCube::MeshType() const
{
	return VoxelObjectType::DYNAMIC;
}