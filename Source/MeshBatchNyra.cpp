#include "MeshBatchNyra.h"
#include "IEUtility/IEMath.h"
#include "IEUtility/IEVector3.h"



const IEVector3 MeshBatchNyra::initalPosBase = IEVector3(0.0f, 0.0f, -4.33f);
const IEVector3 MeshBatchNyra::maxDistance = IEVector3(0.0f, 0.0f, 220.0f);
const IEVector3 MeshBatchNyra::velocityBase = IEVector3(0.0f, 0.0f, 25.0f);
const IEQuaternion MeshBatchNyra::initalOrientation = IEQuaternion(IEMath::ToRadians(270), IEVector3::Yaxis);

// Constructors & Destructor
MeshBatchNyra::MeshBatchNyra(const char* sceneFileName,
							 float minVoxSpan,
							 IEVector3 velocity,
							 IEVector3 initalPos)
	: MeshBatchSkeletal(sceneFileName, minVoxSpan)
	, currentPos(IEVector3::ZeroVector)
	, currentOrientation(initalOrientation)
	, velocity(velocity)
	, initalPos(initalPos)
{}

// Interface
void MeshBatchNyra::Update(double elapsedS)
{
	MeshBatchSkeletal::Update(elapsedS);
	currentPos += velocity * static_cast<float>(elapsedS);

	if((currentPos - maxDistance).Length() < 10.0f)
	{
		currentOrientation *= IEQuaternion(IEMath::ToRadians(180), IEVector3::Yaxis);
		currentPos = -maxDistance;
	}

	// Animate over Hip
	for(uint32_t modelId = 1; modelId < batchDrawParams.getModelTransformBuffer().Count(); modelId++)
	{
		IEMatrix4x4 transform = IEMatrix4x4::Rotate(currentOrientation);
		batchDrawParams.getModelTransformBuffer().CPUData()[modelId].modelRotation = transform;
		
		transform = IEMatrix4x4::Translate(initalPos) * transform * IEMatrix4x4::Translate(currentPos);
		batchDrawParams.getModelTransformBuffer().CPUData()[modelId].model = transform;
	}
	batchDrawParams.getModelTransformBuffer().SendData();
}