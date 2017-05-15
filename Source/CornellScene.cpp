#include "CornellScene.h"
#include "BatchFunctors.h"


CornellScene::CornellScene(const std::string& name,
						   const std::vector<std::string>& rigidFileNames,
						   const std::vector<std::string>& skeletalFileNames,
						   const std::vector<Light>& lights)
	: ConstantScene(name, rigidFileNames,
					skeletalFileNames,
					lights)
{}

void CornellScene::Update(double elapsedS)
{
	// only one object (cube) rotate it from both directions
	static constexpr float sphereSpeedSlow = 70.123f;
	static constexpr float sphereSpeedFast = 113.123f;

	DrawBuffer& dBuffer = rigidBatch.getDrawBuffer();
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(2), sphereSpeedSlow * elapsedS, IEVector3::YAxis);
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(3), sphereSpeedFast * elapsedS, IEVector3::YAxis);
	
	dBuffer.SendModelTransformToGPU(2, 2);	
	ConstantScene::Update(elapsedS);
}