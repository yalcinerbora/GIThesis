#include "MeshBatchSponza.h"
#include "BatchFunctors.h"

MeshBatchSponza::MeshBatchSponza(const char* sceneFileName,
								 float minVoxSpan,
								 const Array32<size_t> maxVoxelCounts)
	: MeshBatchStatic(sceneFileName, minVoxSpan, maxVoxelCounts)
{}

// Static Files
const char* MeshBatchSponza::sponzaDynamicFileName = "sponzaDynamic.gfg";
size_t MeshBatchSponza::sponzaDynamicVoxelSizes[] =
{
	static_cast<size_t>(1024 * 30.0f),
	static_cast<size_t>(1024 * 10.0f),
	static_cast<size_t>(1024 * 5.0f)
};

void MeshBatchSponza::Update(double elapsedS)
{
	// Static Indexing Etc Yolo
	// This ordering may change if maya gfg exporter decides to traverse DF differently
	// but w/e
	static const uint32_t sphereBlue = 1;
	static const uint32_t cubeRGB = 2;
	static const uint32_t torusSmall = 3;
	static const uint32_t torusMid = 4;
	static const uint32_t torusLarge = 5;
	static const uint32_t sphereGreen = 6;
	static const uint32_t sphereRed = 7;
	static const uint32_t cubeRGW = 8;

	std::vector<ModelTransform>& mtBuff = batchDrawParams.getModelTransformBuffer().CPUData();

	BatchFunctors::ApplyRotation rotationFunctor(mtBuff);
	BatchFunctors::ApplyTranslation translationFunctor(mtBuff);

	// Rotation
	// Torus Rotation (Degrees per second)
	static const float torusSmallSpeed = 90.5f;
	static const float torusMidSpeed = 50.33f;
	static const float torusLargeSpeed = 33.25f;
	rotationFunctor(torusSmall, torusSmallSpeed * elapsedS, IEVector3::Xaxis);
	rotationFunctor(torusMid, torusMidSpeed * elapsedS, IEVector3::Zaxis);
	rotationFunctor(torusLarge, torusLargeSpeed * elapsedS, IEVector3::Zaxis);

	//
	rotationFunctor(sphereGreen, torusLargeSpeed * elapsedS, IEVector3::Yaxis);

	// Cube Rotation
	static const float cubeSpeedRGB = 130.123f;
	static const float cubeSpeedRGW = 100.123f;
	rotationFunctor(cubeRGB, cubeSpeedRGB * elapsedS, IEVector3::Xaxis);
	rotationFunctor(cubeRGB, cubeSpeedRGB * elapsedS, IEVector3::Yaxis);
	rotationFunctor(cubeRGW, cubeSpeedRGW * elapsedS, IEVector3::Xaxis);
	rotationFunctor(cubeRGW, cubeSpeedRGW * elapsedS, IEVector3::Yaxis);

	// Translation
	// Torus Translation
	static const IEVector3 torusDelta = {10.0f, 0.0f, 0.0f};
	//ApplyTranslation(torusSmall, elapsedS, torusDelta);
	//ApplyTranslation(torusMid, elapsedS, torusDelta);
	//ApplyTranslation(torusLarge, elapsedS, torusDelta);

	// TODO translation (patrol translation requires state)

	batchDrawParams.getModelTransformBuffer().SendData();
}

VoxelObjectType MeshBatchSponza::MeshType() const
{
	return VoxelObjectType::DYNAMIC;
}