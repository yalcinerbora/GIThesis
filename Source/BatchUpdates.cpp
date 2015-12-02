#include "BatchUpdates.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEMath.h"

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

	auto ApplyRotation = [&mtBuff](uint32_t index, double angle, const IEVector3& axis)
	{
		float angleF = static_cast<float>(angle);
		IEMatrix4x4 rot = IEMatrix4x4::Rotate(IEQuaternion(IEMath::ToRadians(angleF), axis));
		mtBuff[index].model = mtBuff[index].model * rot;
		mtBuff[index].modelRotation = mtBuff[index].modelRotation * rot;
	};
	auto ApplyTranslation = [&mtBuff](uint32_t index, 
									  double elapsedS, 
									  const IEVector3& velocity,
									  const IEVector3& min,
									  const IEVector3& max)
	{
		IEMatrix4x4 trans = IEMatrix4x4::Translate(velocity * static_cast<float>(elapsedS));
		mtBuff[index].model = trans * mtBuff[index].model;
	};

	// Rotation
	// Torus Rotation (Degrees per second)
	static const float torusSmallSpeed = 90.5f;
	static const float torusMidSpeed = 50.33f;
	static const float torusLargeSpeed = 33.25f;
	ApplyRotation(torusSmall, torusSmallSpeed * elapsedS, IEVector3::Xaxis);
	ApplyRotation(torusMid, torusMidSpeed * elapsedS, IEVector3::Zaxis);
	ApplyRotation(torusLarge, torusLargeSpeed * elapsedS, IEVector3::Zaxis);

	// Cube Rotation
	static const float cubeSpeedRGB = 130.123f;
	static const float cubeSpeedRGW = 100.123f;
	ApplyRotation(cubeRGB, cubeSpeedRGB * elapsedS, IEVector3::Xaxis);
	ApplyRotation(cubeRGB, cubeSpeedRGB * elapsedS, IEVector3::Yaxis);
	ApplyRotation(cubeRGW, cubeSpeedRGW * elapsedS, IEVector3::Xaxis);
	ApplyRotation(cubeRGW, cubeSpeedRGW * elapsedS, IEVector3::Yaxis);
		
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

}

void BatchUpdates::CubeUpdate(GPUBuffer& vertBuff, DrawBuffer& drawBuff,
							  double elapsedS)
{

}
