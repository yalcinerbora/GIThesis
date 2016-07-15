#include "MeshBatchSkeletal.h"

// Static Files
//const char* MeshBatchSkeletal::tinmanFileName = "tinman.gfg";
const char* MeshBatchSkeletal::tinmanFileName = "snake.gfg";

size_t MeshBatchSkeletal::tinmanVoxelSizes[] =
{
	static_cast<size_t>(1024 * 12),
	static_cast<size_t>(1024 * 5),
	static_cast<size_t>(1024 * 2)
};

MeshBatchSkeletal::MeshBatchSkeletal(const char* sceneFileName,
									 float minVoxSpan,
									 const Array32<size_t> maxVoxelCounts,
									 const Animator& animator)
	: MeshBatch(sceneFileName, minVoxSpan, maxVoxelCounts, true)
	, animator(animator)
	, animKeys(64 * 8)
	, bindPose(64)
	, jointHierarchy(64)
	, curentJointValue(64)
	, finalTransforms(64)
{}

void MeshBatchSkeletal::Update(double elapsedS)
{
	animator.Update(*this);
}

VoxelObjectType MeshBatchSkeletal::MeshType() const
{
	return VoxelObjectType::SKEL_DYNAMIC;
}