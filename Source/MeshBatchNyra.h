/**


*/
#ifndef __MESHBATCHNYRA_H__
#define __MESHBATCHNYRA_H__

#include "MeshBatchSkeletal.h"

class MeshBatchNyra : public MeshBatchSkeletal
{
	private:
		static const IEVector3		maxDistance;
		static const IEQuaternion	initalOrientation;


		IEVector3					currentPos;
		IEQuaternion				currentOrientation;
		IEVector3					initalPos;
		IEVector3					velocity;

	protected:

	public:
		static const IEVector3		initalPosBase;
		static const IEVector3		velocityBase;

		// Constructors & Destructor
								MeshBatchNyra(const char* sceneFileName,
											  float minVoxSpan,
											  IEVector3 velocity = velocityBase,
											  IEVector3 initalPos = initalPosBase);

		// Interface
		void					Update(double elapsedS) override;
};

#endif //__MESHBATCHNYRA_H__