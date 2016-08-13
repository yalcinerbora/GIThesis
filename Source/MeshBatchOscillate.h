/**


*/
#ifndef __MESHBATCHOSCILLATE_H__
#define __MESHBATCHOSCILLATE_H__

#include "MeshBatch.h"

class MeshBatchOscillate : public MeshBatch
{
	private:
		std::vector<IEVector3>		oscillateParams;
		float						totalTimeS;
		std::vector<ModelTransform>	baseModel;
		IEVector3					oscillateAxis;

		static const float			oscillationAmp;
		static const float			oscillationSpeed;

		static const float			oscillationAmpVariance;
		static const float			oscillationSpeedVariance;

	protected:

	public:
		// Constructors & Destructor
								MeshBatchOscillate(const char* sceneFileName,
												   float minVoxSpan,
												   const IEVector3& oscillateAxis);

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};
#endif //__MESHBATCHOSCILLATE_H__