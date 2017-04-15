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
		StructuredBuffer<ModelTransform>	finalTransforms;
		AnimationBatch						animations;

	public:
		// Constructors & Destructor
											MeshBatchSkeletal();
											MeshBatchSkeletal(const std::vector<VertexElement>& vertexDefintion, uint32_t byteStride,
													  const std::vector<std::string>& sceneFiles);
											MeshBatchSkeletal(const MeshBatchSkeletal&) = delete;
											MeshBatchSkeletal(MeshBatchSkeletal&&);
		MeshBatchSkeletal&					operator=(MeshBatchSkeletal&&);
		MeshBatchSkeletal&					operator=(const MeshBatchSkeletal&) = delete;
											~MeshBatchSkeletal() = default;

		// Interface
		void								Update(double elapsedS) override;
		MeshBatchType						MeshType() const override;
		
		StructuredBuffer<ModelTransform>&	getJointTransforms();
};

#endif //__MESHBATCHSTATIC_H__