/**


*/
#ifndef __BATCH_FUNCTORS_H__
#define __BATCH_FUNCTORS_H__

#include "DrawBuffer.h"
#include "IEUtility/IEMath.h"
#include "IEUtility/IEQuaternion.h"
#include <cstdint>
#include <vector>

using MtBuffer = std::vector<ModelTransform>&;

namespace BatchFunctors
{

	inline void ApplyTranslation(ModelTransform& mt,
						  double elapsedS,
						  IEVector3& velocity)
	{
		IEMatrix4x4 trans = IEMatrix4x4::Translate(velocity * static_cast<float>(elapsedS));
		mt.model = trans * mt.model;
	}

	inline void ApplyRotation(ModelTransform& mt,
					   double angle,
					   const IEVector3& axis)
	{
		float angleF = static_cast<float>(angle);
		IEMatrix4x4 rot = IEMatrix4x4::Rotate(IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef) * angleF,
														   axis));
		mt.model = mt.model * rot;
		mt.modelRotation = mt.modelRotation * rot;
	}
}
#endif //__BATCH_FUNCTORS_H__