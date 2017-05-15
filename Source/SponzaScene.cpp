#include "SponzaScene.h"
#include "BatchFunctors.h"

const IEQuaternion SponzaScene::initalOrientation = IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef) * 270.0f,
																 IEVector3::YAxis);

SponzaScene::SponzaScene(const std::string& name,
						 const std::vector<std::string>& rigidFileNames,
						 const std::vector<std::string>& skeletalFileNames,
						 const std::vector<Light>& lights)
	: ConstantScene(name, rigidFileNames,
					skeletalFileNames,
					lights)
	, velocity(velocityBase)
	, initalPos(initalPosBase)
	, currentOrientation(initalOrientation)
{}

void SponzaScene::Initialize()
{
	skeletalBatch.getAnimationBatch().ChangeAnimationParams(0, 0.0f, 0.5f, AnimationType::REPEAT);
}

void SponzaScene::PatrolNyra(double elapsedS)
{
	static constexpr IEVector3 maxDistance = IEVector3(0.0f, 0.0f, 220.0f);

	DrawBuffer& dBuffer = skeletalBatch.getDrawBuffer();
	currentPos += velocity * static_cast<float>(elapsedS);

	if((currentPos - maxDistance).Length() < 10.0f)
	{
		currentOrientation *= IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef) * 180.0f,
										   IEVector3::YAxis);
		currentPos = -maxDistance;
	}

	// Animate over Hip
	for(uint32_t modelId = 1; modelId < dBuffer.getModelTransformCount(); modelId++)
	{
		IEMatrix4x4 transform = IEMatrix4x4::Rotate(currentOrientation);

		auto& mt = dBuffer.getModelTransform(modelId);
		mt.modelRotation = transform;

		transform = IEMatrix4x4::Translate(initalPos) * transform * IEMatrix4x4::Translate(currentPos);
		mt.model = transform;
	}
	dBuffer.SendModelTransformToGPU();
}

void SponzaScene::Update(double elapsedS)
{
	// Static Indexing Etc Yolo
	// This ordering may change if maya gfg exporter decides to traverse DF differently
	// but w/e
	static constexpr uint32_t dynamicGFGOffset = 382;
	static constexpr uint32_t sphereBlue = dynamicGFGOffset + 1;
	static constexpr uint32_t cubeRGB = dynamicGFGOffset + 2;
	static constexpr uint32_t torusSmall = dynamicGFGOffset + 3;
	static constexpr uint32_t torusMid = dynamicGFGOffset + 4;
	static constexpr uint32_t torusLarge = dynamicGFGOffset + 5;
	static constexpr uint32_t sphereGreen = dynamicGFGOffset + 6;
	static constexpr uint32_t sphereRed = dynamicGFGOffset + 7;
	static constexpr uint32_t cubeRGW = dynamicGFGOffset + 8;

	DrawBuffer& dBuffer = rigidBatch.getDrawBuffer();

	// Rotation
	// Torus Rotation (Degrees per second)
	static const float torusSmallSpeed = 90.5f;
	static const float torusMidSpeed = 50.33f;
	static const float torusLargeSpeed = 33.25f;
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(torusSmall), torusSmallSpeed * elapsedS, IEVector3::XAxis);
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(torusMid), torusMidSpeed * elapsedS, IEVector3::ZAxis);
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(torusLarge), torusLargeSpeed * elapsedS, IEVector3::ZAxis);

	// Rotating Sphere
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(sphereGreen), torusLargeSpeed * elapsedS, IEVector3::YAxis);

	// Cube Rotation
	static const float cubeSpeedRGB = 130.123f;
	static const float cubeSpeedRGW = 1.13f;
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(cubeRGB), cubeSpeedRGB * elapsedS, IEVector3::XAxis);
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(cubeRGB), cubeSpeedRGB * elapsedS, IEVector3::YAxis);
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(cubeRGW), cubeSpeedRGW * elapsedS, IEVector3::XAxis);
	BatchFunctors::ApplyRotation(dBuffer.getModelTransform(cubeRGW), cubeSpeedRGW * elapsedS, IEVector3::YAxis);

	dBuffer.SendModelTransformToGPU(dynamicGFGOffset + 1, 8);

	// Nyra Patrol Animation
	PatrolNyra(elapsedS);

	ConstantScene::Update(elapsedS);
}