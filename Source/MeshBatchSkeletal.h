/**


*/
#ifndef __MESHBATCHSKELETAL_H__
#define __MESHBATCHSKELETAL_H__

#include "MeshBatch.h"
#include "GFG/GFGHeader.h"
#include "Animator.h"

class MeshBatchSkeletal : public MeshBatch
{
	private:
		
	protected:
		StructuredBuffer<IEVector4>		animKeys;
		StructuredBuffer<GFGTransform>  bindPose;
		StructuredBuffer<uint32_t>		jointHierarchy;
		StructuredBuffer<IEVector4>		curentJointValue;
		StructuredBuffer<IEMatrix4x4>	finalTransforms;

		const Animator&					animator;

	public:
		// Constructors & Destructor
								MeshBatchSkeletal(const char* sceneFileName,
												  float minVoxSpan,
												  const Array32<size_t> maxVoxelCounts,
												  const Animator& animator);
	  
		// Static Files
		static const char*		tinmanFileName;

		static size_t			tinmanVoxelSizes[];
  
		// Interface
		void					Update(double elapsedS) override;

		VoxelObjectType			MeshType() const override;
};

#endif //__MESHBATCHSTATIC_H__