/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include <vector>
#include <list>
#include "SolutionI.h"
#include "Shader.h"
#include "FrameTransformBuffer.h"
#include "GICudaVoxelScene.h"
#include "StructuredBuffer.h"
#include "IEUtility/IEVector3.h"
#include "VoxelRenderTexture.h"
#include "VoxelDebugVAO.h"

#pragma pack(push, 1)
struct ObjGridInfo
{
	float span;
	uint32_t voxCount;
};

struct VoxelData
{
	uint32_t vox[2];
};

struct VoxelRenderData
{
	IEVector3 normal;
	uint32_t color;
};
#pragma pack(pop)

class ThesisSolution : public SolutionI
{
	private:
		SceneI*					currentScene;

		Shader					vertexDebugVoxel;
		Shader					fragmentDebugVoxel;

		Shader					vertexVoxelizeObject;
		Shader					geomVoxelizeObject;
		Shader					fragmentVoxelizeObject;
		Shader					computeVoxelizeCount;
		Shader					computePackObjectVoxels;
		Shader					computeDetermineVoxSpan;

		FrameTransformBuffer	cameraTransform;

		StructuredBuffer<ObjGridInfo>			objectGridInfo;
		StructuredBuffer<VoxelData>				voxelData;
		StructuredBuffer<VoxelRenderData>		voxelRenderData;
		StructuredBuffer<uint32_t>				voxelCacheUsageSize;
		VoxelDebugVAO							voxelVAO;

		// Cuda Segment
		GICudaVoxelScene		voxelScene;

		static size_t			InitialObjectGridSize;
		static size_t			InitialVoxelBufferSizes;
		static size_t			MaxVoxelCacheSize;

	protected:
		void					CreateVoxel(size_t index);
		void					DrawVoxel(size_t index);

	public:
								ThesisSolution();
								ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&	operator= (const ThesisSolution&) = delete;
								~ThesisSolution() = default;

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Frame(const Camera&) override;
};
#endif //__THESISSOLUTION_H__