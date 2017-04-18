#include "Animation.h"

AnimationBatch::AnimationBatch(const std::vector<std::string>& fileNames)
{}

void AnimationBatch::GenerateFinalTransforms(ModelTransform out[], double time)
{

}

void AnimationBatch::GenerateFinalTransforms(ModelTransform out[],
											 double time,
											 size_t animationOffset,
											 size_t animationCount)
{

}

void AnimationBatch::AnimationParams(uint32_t animationIndex,
									 float delay,
									 float speedMod,
									 AnimationType type)
{

}


//GI_LOG("Animation load complete");
//GI_LOG("\tDuration : %f ms", timer.ElapsedMilliS());
//GI_LOG("\tKey Count : %d", header.keyCount);
//GI_LOG("\tJoint Count : %zd", jointKeys.size());
//GI_LOG("----------");


//
//
//
//
//GenerateFinalTransforms()
//{
//	uint32_t keyFrom, keyTo;
//	float weight;
//	FindKeys(keyFrom, keyTo, weight, static_cast<float>(elapsedS));
//	UpdateAnimMatrices(keyFrom, keyTo, weight);
//	finalTransforms.SendData();
//}

//void MeshBatchSkeletal::GenInvBindMatrix()
//{
//	invBindPose.resize(jointHierarchy.size());
//	for(uint32_t boneId = 0; boneId < jointHierarchy.size(); boneId++)
//	{
//		IEMatrix4x4 transform, rotation;
//		for(unsigned int i = boneId; i != 0xFFFFFFFF; i = jointHierarchy[i])
//		{
//			IEMatrix4x4 trans, rot;
//			MeshBatch::GenTransformMatrix(trans, rot, bindPose[i]);
//			//trans.InverseSelf();
//			//rot.InverseSelf();
//
//			transform = trans * transform;
//			rotation = rot * rotation;
//		}
//		invBindPose[boneId].model = transform.Inverse();
//		invBindPose[boneId].modelRotation = rotation.Inverse();;
//	}
//}
//
//void  MeshBatchSkeletal::FindKeys(uint32_t& keyFrom,
//								  uint32_t& keyTo,
//								  float& weight,
//								  float elapsedS)
//{
//	// Test version clamp to next animation
//	// skip interpolation
//	timeS += elapsedS;
//	if(type == AnimationType::ONCE && timeS > keyTimes.back())
//	{
//		keyFrom = 0;
//		keyTo = 0;
//		weight = 0.0f;
//		return;
//	}
//
//	if(timeS > keyTimes.back()) animState = !animState;
//	timeS = std::fmod(timeS, keyTimes.back());
//
//	// Time value
//	float time = timeS;
//	if(type == AnimationType::OSCILLATE && animState == true)
//		time = keyTimes.back() - timeS;
//
//	// Find nearest frame
//	// TODO: Look for better algo here O(n) looks kinda dumb
//	keyFrom = static_cast<uint32_t>(keyTimes.size() - 1);
//	keyTo = 0;
//	for(unsigned int i = 0; i < keyTimes.size(); i++)
//	{
//		if((time > keyTimes[i] &&
//			(i != keyTimes.size() - 1 && time < keyTimes[i + 1])))
//		{
//			keyFrom = i;
//			keyTo = i + 1;
//		}
//	}
//	if(keyFrom == keyTimes.size() - 1)
//		weight = (time / keyTimes[keyTo]);
//	else
//		weight = (time - keyTimes[keyFrom]) / (keyTimes[keyTo] - keyTimes[keyFrom]);
//	assert(weight >= 0.0f && weight <= 1.0f);
//	//GI_LOG("Time %f, KeyFromTo {%d, %d}, Weight %f", time, keyFrom, keyTo, weight);
//}
//
//void MeshBatchSkeletal::UpdateAnimMatrices(uint32_t keyFrom, uint32_t keyTo, float weight)
//{
//	// Interp Hip Translate
//	IEVector3 hipLerp = IEFunctions::Lerp(hipTranslations[keyFrom], hipTranslations[keyTo], weight);
//
//	//GI_DEBUG_LOG("Hip Lerp %f, %f, %f", hipLerp.getX(), hipLerp.getY(), hipLerp.getZ());
//
//	// TODO Atm there is redundant work (down top approach parallel friendly)
//	for(uint32_t boneId = 0; boneId < jointHierarchy.size(); boneId++)
//	{
//		IEMatrix4x4 transform, rotation;
//		for(unsigned int i = boneId; i != 0xFFFFFFFF; i = jointHierarchy[i])
//		{
//			bool isHip = (jointHierarchy[i] == 0xFFFFFFFF);
//
//			// TODO Interpolate
//			IEQuaternion from = jointKeys[i][keyFrom];
//			IEQuaternion to = jointKeys[i][keyTo];
//			IEQuaternion interp = IEQuaternion::SLerp(from, to, weight);
//			//IEQuaternion interp = from;
//
//			// Generate Transformation Matrix
//			IEMatrix4x4 trans, rot;
//			GenTransformMatrix(trans, rot,
//				(isHip) ? hipLerp : IEVector3(bindPose[i].translate),
//							   IEVector3(bindPose[i].scale),
//							   interp);
//
//			transform = trans * transform;
//			rotation = rot * rotation;
//		}
//		finalTransforms.CPUData()[boneId].model = transform * invBindPose[boneId].model;
//		finalTransforms.CPUData()[boneId].modelRotation = rotation * invBindPose[boneId].modelRotation;
//	}
//}
//
//void  MeshBatchSkeletal::AnimationParams(float delay, float speedMod, AnimationType type)
//{
//	assert(delay >= 0.0f);
//	this->type = type;
//	this->delay = delay;
//	this->speedMod = speedMod;
//
//	float speedInv = 1.0f / speedMod;
//	for(float& time : keyTimes) time = time * speedInv + delay;
//}
//
//void MeshBatchSkeletal::GenTransformMatrix(IEMatrix4x4& transform,
//										   IEMatrix4x4& rotation,
//										   const IEVector3& translate,
//										   const IEVector3& scale,
//										   const IEQuaternion& rotationQuat)
//{
//	rotation = IEMatrix4x4::Rotate(rotationQuat);
//
//	transform = IEMatrix4x4::Scale(scale.getX(), scale.getY(), scale.getZ()) * rotation;
//	transform = IEMatrix4x4::Translate(translate) * transform;
//}