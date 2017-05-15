#include "Animation.h"
#include "GFGLoader.h"
#include "Macros.h"
#include "MeshBatch.h"
#include "IEUtility/IETimer.h"
#include "IEUtility/IEFunctions.h"

AnimationBatch::AnimationBatch(const std::vector<std::string>& fileNames)
{
	IETimer timer;
	timer.Start();
	for(const std::string& fName : fileNames)
	{
		GFGLoader::LoadAnim(*this, fName.c_str());
		GI_LOG("Animation load from \"%s\" complete", fName.c_str());		
	}
	timer.Stop();
	GI_LOG("Total Duration : %f ms", timer.ElapsedMilliS());
	GI_LOG("Animations");
	for(int i = 0; i < animations.size(); i++)
	{
		GI_LOG("----------");
		GI_LOG("\t%s", (std::string("Animation#") + std::to_string(i)).c_str());
		GI_LOG("\tKey Count   : %d", animations[i].keyCount);
		GI_LOG("\tJoint Count : %d", animations[i].jointCount);		
	}
}

void AnimationBatch::ChangeAnimationParams(uint32_t animationIndex,
										   float delay,
										   float speedMod,
										   AnimationType type)
{
	Animation& anim = animations[animationIndex];

	assert(delay >= 0.0f);
	anim.delay = delay;
	anim.speedMod = speedMod;
	anim.type = type;

	float speedInv = 1.0f / speedMod;
	
	for(uint32_t i = anim.keyHipOffset;
		i < anim.keyHipOffset + anim.keyCount;
		i++)
	{
		keyTimes[i] = keyTimes[i] * speedInv + delay;
	}		
}

void AnimationBatch::LoadAnimation(const std::vector<float>& animKeyTimes,
								   const std::vector<IEVector3>& animHipTranslations,
								   const std::vector<IEQuaternion>& animJointRotations,
								   const std::vector<GFGTransform>& animBindPose,
								   const std::vector<ModelTransform>& animInvBindPose,
								   const std::vector<uint32_t>& animJointHierarchies,
								   uint32_t keyCount,
								   uint32_t jointCount)
{
	Animation newAnim = {};
	newAnim.jointCount = jointCount;
	newAnim.keyCount = keyCount;

	// Default Play Parameters
	newAnim.state = false;
	newAnim.delay = 0.0f;
	newAnim.speedMod = 1.0f;
	newAnim.type = AnimationType::REPEAT;

	// Offsets
	newAnim.jointOffset = static_cast<uint32_t>(invBindPoses.size());
	newAnim.rotationOffset = static_cast<uint32_t>(jointRotations.size());
	newAnim.keyHipOffset = static_cast<uint32_t>(keyTimes.size());

	keyTimes.insert(keyTimes.end(), animKeyTimes.begin(), animKeyTimes.end());
	hipTranslations.insert(hipTranslations.end(), animHipTranslations.begin(), animHipTranslations.end());
	jointRotations.insert(jointRotations.end(), animJointRotations.begin(), animJointRotations.end());
	bindPose.insert(bindPose.end(), animBindPose.begin(), animBindPose.end());
	invBindPoses.insert(invBindPoses.end(), animInvBindPose.begin(), animInvBindPose.end());
	
	// Load Hierarchy and adjust pointer hierarchy
	jointHierarchies.insert(jointHierarchies.end(), animJointHierarchies.begin(), animJointHierarchies.end());
	for(uint32_t i = newAnim.jointOffset; i < newAnim.jointOffset + jointCount; i++)
	{
		if(jointHierarchies[i] != std::numeric_limits<uint32_t>::max())
		{
			jointHierarchies[i] += newAnim.jointOffset;
		}
	}
	// Finally Push to the vector
	animations.push_back(newAnim);
}

void  AnimationBatch::FindKeys(uint32_t& localKeyFrom,
							   uint32_t& localKeyTo,
							   float& weight,
							   uint32_t animationIndex,
							   float elapsedS)
{
	Animation& anim = animations[animationIndex];
	float lastTime = keyTimes[anim.keyHipOffset + anim.keyCount - 1];
	// Test version clamp to next animation
	// skip interpolation
	anim.timeS += elapsedS;
	if(anim.type == AnimationType::ONCE &&
	   anim.timeS > lastTime)
	{
		localKeyFrom = 0;
		localKeyTo = 0;
		weight = 0.0f;
		return;
	}

	if(anim.timeS > lastTime)
	{
		anim.state = !anim.state;
		anim.timeS = std::fmod(anim.timeS, lastTime);
	}
	
	// Time value
	float time = anim.timeS;
	if(anim.type == AnimationType::OSCILLATE && anim.state == true)
		time = keyTimes.back() - anim.timeS;

	// Find nearest frame
	// TODO: Look for better algo here O(n) looks kinda dumb
	uint32_t timeFirst = anim.keyHipOffset;
	uint32_t timeLast = anim.keyHipOffset + anim.keyCount;

	uint32_t batchKeyFrom = timeLast - 1;
	uint32_t batchKeyTo = timeFirst;
	for(uint32_t i = timeFirst; i < timeLast - 1; i++)
	{
		// Check if current elapsed time is in between those two frames
		if(time > keyTimes[i] &&
		   time < keyTimes[i + 1])
		{
			batchKeyFrom = i;
			batchKeyTo = i + 1;
		}
	}

	// Determine Interpolation weight
	if(batchKeyFrom == timeLast - 1)
	{
		weight = (time / keyTimes[batchKeyTo]);
	}
	else
	{
		weight = (time - keyTimes[batchKeyFrom]) / (keyTimes[batchKeyTo] - keyTimes[batchKeyFrom]);
	}
	assert(weight >= 0.0f && weight <= 1.0f);

	localKeyFrom = batchKeyFrom - anim.jointOffset;
	localKeyTo = batchKeyTo - anim.jointOffset;
	//GI_LOG("Time %f, KeyFromTo {%d, %d}, Weight %f", time, localKeyFrom, localKeyTo, weight);
}

void AnimationBatch::GenerateAnimMatrices(ModelTransform newTransforms[],
										  uint32_t localKeyFrom,
										  uint32_t localKeyTo,
										  float weight,
										  uint32_t animationIndex)
{
	const Animation& anim = animations[animationIndex];

	// Interp Hip Translate	
	uint32_t batchKeyFrom = localKeyFrom + anim.jointOffset;
	uint32_t batchKeyTo = localKeyTo + anim.jointOffset;
	IEVector3 hipLerp = IEFunctions::Lerp(hipTranslations[batchKeyFrom],
										  hipTranslations[batchKeyTo],
										  weight);

	//GI_DEBUG_LOG("Hip Lerp %f, %f, %f", hipLerp.getX(), hipLerp.getY(), hipLerp.getZ());

	// TODO Atm there is redundant work (down top approach parallel friendly)
	for(uint32_t localBoneId = 0; localBoneId < anim.jointCount; localBoneId++)
	{
		uint32_t batchBoneId = localBoneId + anim.jointOffset;

		IEMatrix4x4 transform, rotation;
		for(uint32_t i = batchBoneId; i != 0xFFFFFFFF; i = jointHierarchies[i])
		{
			bool isHip = (jointHierarchies[i] == 0xFFFFFFFF);

			// Rotation Interpolation
			uint32_t localJointFrom = i * anim.keyCount + localKeyFrom;
			uint32_t localJointTo = i * anim.keyCount + localKeyTo;

			IEQuaternion from = jointRotations[anim.rotationOffset + localJointFrom];
			IEQuaternion to = jointRotations[anim.rotationOffset + localJointTo];
			IEQuaternion quatInterp = IEQuaternion::SLerp(from, to, weight);

			// Translation
			const IEVector3& translation = (isHip) ? hipLerp : IEVector3(bindPose[i].translate);
			
			// Generate Transformation Matrix
			IEMatrix4x4 rot = IEMatrix4x4::Rotate(quatInterp);
			IEMatrix4x4 trans = IEMatrix4x4::Translate(translation) *
								IEMatrix4x4::Scale(bindPose[i].scale[0],
												   bindPose[i].scale[1], 
												   bindPose[i].scale[2]) *
								rot;

			transform = trans * transform;
			rotation = rot * rotation;
		}
		newTransforms[localBoneId].model = transform * invBindPoses[localBoneId].model;
		newTransforms[localBoneId].modelRotation = rotation * invBindPoses[localBoneId].modelRotation;
	}
}

void AnimationBatch::UpdateFinalTransforms(ModelTransform newTransforms[],
										   double elapsedS,
										   uint32_t animationIndex)
{
	uint32_t keyFrom, keyTo;
	float weight;
	FindKeys(keyFrom, keyTo, weight, animationIndex, static_cast<float>(elapsedS));
	GenerateAnimMatrices(newTransforms, keyFrom, keyTo, weight, animationIndex);
}

const Animation& AnimationBatch::GetAnimation(uint32_t index) const
{
	return animations[index];
}

uint32_t AnimationBatch::AnimationCount() const
{
	return static_cast<uint32_t>(animations.size());
}

uint32_t AnimationBatch::TotalJointCount() const
{
	return static_cast<uint32_t>(hipTranslations.size());
}