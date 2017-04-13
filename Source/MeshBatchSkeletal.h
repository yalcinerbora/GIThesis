/**


*/
#ifndef __MESHBATCHSKELETAL_H__
#define __MESHBATCHSKELETAL_H__

#include "MeshBatch.h"
#include "GFG/GFGHeader.h"
#include "IEUtility/IEQuaternion.h"

#define JOINT_PER_VERTEX

using JointKeys = std::vector<std::vector<IEQuaternion>>;

enum class AnimationType
{
	OSCILLATE,
	ONCE,
	REPEAT,
};

class MeshBatchSkeletal : public MeshBatch
{
	private:
		
	protected:
		StructuredBuffer<ModelTransform>	finalTransforms;
		
		// From GFG
		GFGAnimationHeader					header;
		std::vector<IEVector3>				hipTranslations;
		std::vector<float>					keyTimes;
		JointKeys							jointKeys;		
		std::vector<GFGTransform>			bindPose;
		std::vector<uint32_t>				jointHierarchy;

		// Generated from GFG
		std::vector<ModelTransform>			invBindPose;

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

		static void							GenTransformMatrix(IEMatrix4x4& transform,
															   IEMatrix4x4& rotation,
															   const IEVector3& translate,
															   const IEVector3& scale,
															   const IEQuaternion& rotationQuat);

	public:
		// Constructors & Destructor
								MeshBatchSkeletal(const std::vector<const VertexElement>& vertexDefintion,
												  uint32_t byteStride,
												  const std::vector<std::string>& sceneFiles);
	  
		// Static Files
		static const char*					tinmanFileName;
		static const char*					nyraFileName;
		static const char*					snakeFileName;

		static size_t						tinmanVoxelSizes[];
  
		// Interface
		void								Update(double elapsedS) override;

		VoxelObjectType						MeshType() const override;
		void								AnimationParams(float delay,
															float speedMod,
															AnimationType type);

		StructuredBuffer<ModelTransform>&	getJointTransforms();
};

#endif //__MESHBATCHSTATIC_H__