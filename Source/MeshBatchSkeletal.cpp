#include "MeshBatchSkeletal.h"
#include "GFGLoader.h"
#include "Macros.h"
#include "IEUtility/IETimer.h"
#include "IEUtility/IEFunctions.h"

MeshBatchSkeletal::MeshBatchSkeletal()
	: MeshBatch()
{}

MeshBatchSkeletal::MeshBatchSkeletal(const std::vector<VertexElement>& vertexDefintion,
									 uint32_t byteStride,
									 const std::vector<std::string>& sceneFiles)
	: MeshBatch(vertexDefintion, byteStride, sceneFiles)
	, animations(sceneFiles)
{}

MeshBatchSkeletal::MeshBatchSkeletal(MeshBatchSkeletal&& other)
	: MeshBatch(std::move(other))
	, finalTransforms(std::move(other.finalTransforms))
	, animations(std::move(other.animations))
{}

MeshBatchSkeletal& MeshBatchSkeletal::operator=(MeshBatchSkeletal&& other)
{
	assert(this != &other);
	finalTransforms = std::move(other.finalTransforms);
	animations = std::move(other.animations);
	return *this;
}

void MeshBatchSkeletal::Update(double elapsedS)
{
}

MeshBatchType MeshBatchSkeletal::MeshType() const
{
	return MeshBatchType::SKELETAL;
}

StructuredBuffer<ModelTransform>& MeshBatchSkeletal::getJointTransforms()
{
	return finalTransforms;
}