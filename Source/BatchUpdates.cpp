#include "BatchUpdates.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEMath.h"

class ApplyTranslation
{
	using MtBuffer = std::vector<ModelTransform>&;
	private: MtBuffer mtBuffer;
	public: 
		ApplyTranslation(MtBuffer buff) : mtBuffer(buff) {}
		void operator()(uint32_t index,
						double elapsedS,
						const IEVector3& velocity)
		{
			IEMatrix4x4 trans = IEMatrix4x4::Translate(velocity * static_cast<float>(elapsedS));
			mtBuffer[index].model = trans * mtBuffer[index].model;
		}
};

class ApplyRotation
{
	using MtBuffer = std::vector<ModelTransform>&;
	private: MtBuffer mtBuffer;
	public:
		ApplyRotation(MtBuffer buff) : mtBuffer(buff) {}
		void operator()(uint32_t index,
						double angle,
						const IEVector3& axis)
		{
			float angleF = static_cast<float>(angle);
			IEMatrix4x4 rot = IEMatrix4x4::Rotate(IEQuaternion(IEMath::ToRadians(angleF), axis));
			mtBuffer[index].model = mtBuffer[index].model * rot;
			mtBuffer[index].modelRotation = mtBuffer[index].modelRotation * rot;
		}
};

void BatchUpdates::SponzaUpdate(GPUBuffer& vertBuff, DrawBuffer& drawBuff,
								double elapsedS)
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
	
	std::vector<ModelTransform>& mtBuff = drawBuff.getModelTransformBuffer().CPUData();

	ApplyRotation rotationFunctor(mtBuff);
	ApplyTranslation translationFunctor(mtBuff);

	// Rotation
	// Torus Rotation (Degrees per second)
	static const float torusSmallSpeed = 90.5f;
	static const float torusMidSpeed = 50.33f;
	static const float torusLargeSpeed = 33.25f;
	rotationFunctor(torusSmall, torusSmallSpeed * elapsedS, IEVector3::Xaxis);
	rotationFunctor(torusMid, torusMidSpeed * elapsedS, IEVector3::Zaxis);
	rotationFunctor(torusLarge, torusLargeSpeed * elapsedS, IEVector3::Zaxis);

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

	drawBuff.getModelTransformBuffer().SendData();
}

void BatchUpdates::CornellUpdate(GPUBuffer& vertBuff, DrawBuffer& drawBuff,
								 double elapsedS)
{
	// only one object (cube) rotate it from both directions
	static const float sphereSpeedSlow = 70.123f;
	static const float sphereSpeedFast = 113.123f;
	std::vector<ModelTransform>& mtBuff = drawBuff.getModelTransformBuffer().CPUData();
	ApplyRotation rotationFunctor(mtBuff);
	
	rotationFunctor(1, sphereSpeedSlow * elapsedS, IEVector3::Yaxis);
	rotationFunctor(2, sphereSpeedFast * elapsedS, IEVector3::Yaxis);
	drawBuff.getModelTransformBuffer().SendData();
}

void BatchUpdates::CubeUpdate(GPUBuffer& vertBuff, DrawBuffer& drawBuff,
							  double elapsedS)
{
	// only one object (cube) rotate it from both directions
	static const float cubeSpeedRGB = 130.123f;
	std::vector<ModelTransform>& mtBuff = drawBuff.getModelTransformBuffer().CPUData();
	ApplyRotation rotationFunctor(mtBuff);
	rotationFunctor(1, cubeSpeedRGB * elapsedS, IEVector3::Xaxis);
	rotationFunctor(1, cubeSpeedRGB * elapsedS, IEVector3::Yaxis);
	drawBuff.getModelTransformBuffer().SendData();
}
