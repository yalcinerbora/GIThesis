/**


*/
#ifndef __MESHBATCHSKELETAL_H__
#define __MESHBATCHSKELETAL_H__

#include "MeshBatch.h"
#include "GFG/GFGHeader.h"
#include "IEUtility/IEQuaternion.h"
#include "Animation.h"

#define JOINT_PER_VERTEX

class MeshBatchSkeletal : public MeshBatch
{
	private:
		
	protected:
		AnimationBatch						animBatch;
		StructuredBuffer<ModelTransform>	finalTransforms;

	public:
		// Constructors & Destructor
											MeshBatchSkeletal();
											MeshBatchSkeletal(const std::vector<VertexElement>& vertexDefintion, 
															  uint32_t byteStride,
															  const std::vector<std::string>& sceneFiles,
															  uint32_t repeatCount = 1);
											MeshBatchSkeletal(const MeshBatchSkeletal&) = delete;
											MeshBatchSkeletal(MeshBatchSkeletal&&);
		MeshBatchSkeletal&					operator=(MeshBatchSkeletal&&);
		MeshBatchSkeletal&					operator=(const MeshBatchSkeletal&) = delete;
											~MeshBatchSkeletal() = default;

		// Interface
		void								Update(double elapsedS) override;
		MeshBatchType						MeshType() const override;
		
		StructuredBuffer<ModelTransform>&	getJointTransforms();
		AnimationBatch&						getAnimationBatch();
};

#endif //__MESHBATCHSTATIC_H__