#include "MeshBatchOscillate.h"
#include "BatchFunctors.h"
#include "IEUtility/IERandom.h"

const float MeshBatchOscillate::oscillationAmp = 60.0f;		// Map units
const float MeshBatchOscillate::oscillationAmpVariance = 20.0f;

const float MeshBatchOscillate::oscillationSpeed = 6.0f;	// Map units per second
const float MeshBatchOscillate::oscillationSpeedVariance = 2.5f;

MeshBatchOscillate::MeshBatchOscillate(const char* sceneFileName,
									   float minVoxSpan,
									   const IEVector3& axis)
	: MeshBatch(sceneFileName, minVoxSpan, false)
	, oscillateAxis(axis)
	, totalTimeS(0.0f)
    , oscillationOn(false)
{
	assert(batchDrawParams.getModelTransformBuffer().CPUData().size() == (batchParams.objectCount + 1));
	oscillateParams.resize(batchParams.objectCount);

	IERandom rng;
	for(IEVector3& oscVec : oscillateParams)
	{
		oscVec.setX(static_cast<float>(rng.Double(oscillationAmp, oscillationAmpVariance)));
		oscVec.setY(static_cast<float>(rng.Double(oscillationSpeed, oscillationSpeedVariance)));
	}
	baseModel = batchDrawParams.getModelTransformBuffer().CPUData();
}

void MeshBatchOscillate::Update(double elapsedS)
{
    if(!oscillationOn) return;

	totalTimeS += static_cast<float>(elapsedS);
	std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();

	auto Oscillate = [](ModelTransform& model, float amplitude, float speed, float totalTimeS,
						const ModelTransform& baseModel,
						const IEVector3& oscillateAxis)
	{
		
		float timeToPeakS = amplitude / speed;
		
		// Sinusodial Oscillation
		float oscillation = amplitude * std::sin(totalTimeS * (IEMath::PI / (2.0f * timeToPeakS)));

		IEMatrix4x4 trans = IEMatrix4x4::Translate(oscillateAxis * oscillation);
		model.model = trans * baseModel.model;
	};

    for(unsigned int i = 0; i < mtBuff.size(); i++)
    {
        Oscillate(mtBuff[i],
                    oscillateParams[i].getX(), oscillateParams[i].getY(),
                    totalTimeS,
                    baseModel[i], oscillateAxis);
    }
    batchDrawParams.getModelTransformBuffer().SendData();
}

void MeshBatchOscillate::ToggleOscillate(bool oscillate)
{
    oscillationOn = oscillate;
}

VoxelObjectType MeshBatchOscillate::MeshType() const
{
	return VoxelObjectType::DYNAMIC;
}