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
									 const std::vector<std::string>& sceneFiles,
									 uint32_t repeatCount)
	: MeshBatch(vertexDefintion, byteStride, sceneFiles, repeatCount)
	, animBatch(sceneFiles)
	, finalTransforms(animBatch.TotalJointCount())
{
	if(!sceneFiles.empty())
	{
		for(ModelTransform& t : finalTransforms.CPUData())
		{
			t.model = IEMatrix4x4::IdentityMatrix;
			t.modelRotation = IEMatrix4x4::IdentityMatrix;
		}
		finalTransforms.SendData();
	}
}

MeshBatchSkeletal::MeshBatchSkeletal(MeshBatchSkeletal&& other)
	: MeshBatch(std::move(other))
	, finalTransforms(std::move(other.finalTransforms))
	, animBatch(std::move(other.animBatch))
{}

MeshBatchSkeletal& MeshBatchSkeletal::operator=(MeshBatchSkeletal&& other)
{
	assert(this != &other);
	MeshBatch::operator=(std::move(other));
	finalTransforms = std::move(other.finalTransforms);
	animBatch = std::move(other.animBatch);
	return *this;
}

void MeshBatchSkeletal::Update(double elapsedS)
{
	for(uint32_t i = 0; i < animBatch.AnimationCount(); i++)
	{
		const Animation& anim = animBatch.GetAnimation(i);
		animBatch.UpdateFinalTransforms(finalTransforms.CPUData().data() + anim.jointOffset,
										elapsedS,
										i);
	}	
	if(animBatch.AnimationCount() != 0) finalTransforms.SendData();
}

MeshBatchType MeshBatchSkeletal::MeshType() const
{
	return MeshBatchType::SKELETAL;
}

StructuredBuffer<ModelTransform>& MeshBatchSkeletal::getJointTransforms()
{
	return finalTransforms;
}

AnimationBatch& MeshBatchSkeletal::getAnimationBatch()
{
	return animBatch;
}