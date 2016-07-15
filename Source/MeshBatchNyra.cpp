#include "MeshBatchNyra.h"
#include "BatchFunctors.h"

// Constructors & Destructor
MeshBatchNyra::MeshBatchNyra(const char* sceneFileName,
							 float minVoxSpan,
							 const Array32<size_t> maxVoxelCounts)
	: MeshBatch(sceneFileName, minVoxSpan, maxVoxelCounts, false)
{}

const char* MeshBatchNyra::nyraFileName = "nyra.gfg";
size_t MeshBatchNyra::nyraVoxelSizes[] =
{
	static_cast<size_t>(1024 * 12),
	static_cast<size_t>(1024 * 5),
	static_cast<size_t>(1024 * 2)
};

// Interface
void MeshBatchNyra::Update(double elapsedS)
{
	//// only one object (cube) rotate it from both directions
	//static const float cubeSpeedRGB = 130.123f;
	//std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();
	//BatchFunctors::ApplyRotation rotationFunctor(mtBuff);
	//rotationFunctor(1, cubeSpeedRGB * elapsedS, IEVector3::Yaxis);
	//batchDrawParams.getModelTransformBuffer().SendData();
}

VoxelObjectType MeshBatchNyra::MeshType() const
{
	return VoxelObjectType::DYNAMIC;
}