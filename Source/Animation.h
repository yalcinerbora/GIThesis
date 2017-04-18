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
	GFGAnimationHeader					header;
	uint32_t							jointOffset;
	uint32_t							keyCount;
	float								timeS;
	float								delay;
	float								speedMod;
	AnimationType						type;
	bool								animState;

	//void								GenInvBindMatrix();
	//void								FindKeys(uint32_t& keyFrom,
	//											 uint32_t& keyTo,
	//											 float& weight,
	//											 float elapsedS);
	//void								UpdateAnimMatrices(uint32_t keyFrom,
	//													   uint32_t keyTo,
	//													   float weight);

};

class AnimationBatch
{
	private:
	// From GFG
	std::vector<Animation>					animations;

	std::vector<IEVector3>					hipTranslations;
	std::vector<float>						keyTimes;
	std::vector<std::vector<IEQuaternion>>	jointKeys;
	std::vector<GFGTransform>				bindPoses;
	std::vector<uint32_t>					jointHierarchies;

	// Generated from GFG
	std::vector<ModelTransform>				invBindPoses;

	public:
		// Constructors & Destructor
										AnimationBatch() = default;
										AnimationBatch(const std::vector<std::string>& fileNames);
										~AnimationBatch() = default;

	void								AnimationParams(uint32_t animationIndex,
														float delay,
														float speedMod,
														AnimationType type);
	//void								Update(double elapsedS);
	void								GenerateFinalTransforms(ModelTransform out[], double time);
	void								GenerateFinalTransforms(ModelTransform out[],
																double time,
																size_t animationOffset,
																size_t animationCount);
};
