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

		private: MtBuffer mtBuffer;
		public:
		ApplyRotation(MtBuffer buff) : mtBuffer(buff) {}
		void operator()(uint32_t index,
						double angle,
						const IEVector3& axis)
		{
			float angleF = static_cast<float>(angle);
			IEMatrix4x4 rot = IEMatrix4x4::Rotate(IEQuaternion(static_cast<float>(IEMathConstants::DegToRadCoef) * angleF, 
															   axis));
			mtBuffer[index].model = mtBuffer[index].model * rot;
			mtBuffer[index].modelRotation = mtBuffer[index].modelRotation * rot;
		}
	};
}
#endif //__BATCH_FUNCTORS_H__