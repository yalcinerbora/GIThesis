#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include "Globals.h"
#include "GFG/GFGHeader.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEVector3.h"

enum class AnimationType
{
	OSCILLATE,
	ONCE,
	REPEAT
};

struct Animation
{
	uint32_t							keyCount;
	uint32_t							jointCount;

	uint32_t							jointOffset;
	uint32_t							keyHipOffset;
	uint32_t							rotationOffset;
	
	float								timeS;

	float								delay;
	float								speedMod;
	AnimationType						type;
	bool								state;
};

class AnimationBatch
{
	private:
		std::vector<Animation>			animations;

		std::vector<float>				keyTimes;
		std::vector<IEVector3>			hipTranslations;		
		std::vector<IEQuaternion>		jointRotations;
		std::vector<GFGTransform>		bindPose;
		std::vector<ModelTransform>		invBindPoses;
		std::vector<uint32_t>			jointHierarchies;

		void							FindKeys(uint32_t& localKeyFrom,
												 uint32_t& localKeyTo,
												 float& weight,
												 uint32_t animationIndex,
												 float elapsedS);
		void							GenerateAnimMatrices(ModelTransform newTransforms[],
															 uint32_t localKeyFrom,
															 uint32_t localKeyTo,
															 float weight,
															 uint32_t animationIndex);

	protected:
	public:
		// Constructors & Destructor
										AnimationBatch() = default;
										AnimationBatch(const std::vector<std::string>& fileNames);
										~AnimationBatch() = default;

		// Transformation Matrix Generation
		void							UpdateFinalTransforms(ModelTransform newTransforms[],
															  double elapsedS,
															  uint32_t animationIndex);

		// Load Related
		void							LoadAnimation(const std::vector<float>& keyTimes,
													  const std::vector<IEVector3>& hipTranslations,
													  const std::vector<IEQuaternion>& jointKeys,
													  const std::vector<GFGTransform>& bindPose,
													  const std::vector<ModelTransform>& invBindPose,
													  const std::vector<uint32_t>& localJointHierarchies,
													  uint32_t keyCount,
													  uint32_t jointCount);

		// Change-Set
		const Animation&				GetAnimation(uint32_t index) const;
		uint32_t						AnimationCount() const;
		uint32_t						TotalJointCount() const;
		void							ChangeAnimationParams(uint32_t animationIndex,
															  float delay,
															  float speedMod,
															  AnimationType type);
};
