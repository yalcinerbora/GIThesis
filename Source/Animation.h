#pragma once

#include <vector>
#include "GFG/GFGHeader.h"
#include "IEUtility/IEQuaternion.h"

enum class AnimationType
{
	OSCILLATE,
	ONCE,
	REPEAT
};

struct Animation
{
	GFGAnimationHeader header;
	uint32_t jointOffset;
	uint32_t					keyCount;
	float								timeS;
	float								delay;
	float								speedMod;
	AnimationType						type;
	bool								animState;

	void								GenInvBindMatrix();
	void								FindKeys(uint32_t& keyFrom,
												 uint32_t& keyTo,
												 float& weight,
												 float elapsedS);
	void								UpdateAnimMatrices(uint32_t keyFrom,
														   uint32_t keyTo,
														   float weight);

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

	


	static void							GenTransformMatrix(IEMatrix4x4& transform,
														   IEMatrix4x4& rotation,
														   const IEVector3& translate,
														   const IEVector3& scale,
														   const IEQuaternion& rotationQuat);


	public:
		// Construcots
										AnimationBatch(const std::vector<std::string>& fileNames);
	void								AnimationParams(float delay,
														float speedMod,
														AnimationType type);
										GenerateFinalTransforms(ModelTransform out[], float time);
										GenerateFinalTransforms(ModelTransform out[],
																float time,
																size_t animationOffset,
																size_t animationCount);
};
