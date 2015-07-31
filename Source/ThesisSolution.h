/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include <vector>
#include <list>
#include <AntTweakBar.h>
#include "SolutionI.h"
#include "Shader.h"
#include "FrameTransformBuffer.h"
#include "GICudaVoxelScene.h"
#include "StructuredBuffer.h"
#include "IEUtility/IEVector3.h"
#include "VoxelRenderTexture.h"
#include "VoxelDebugVAO.h"
#include "DrawBuffer.h"

class DeferredRenderer;

#pragma pack(push, 1)
struct ObjGridInfo
{
	float span;
	uint32_t voxCount;
};

struct VoxelData
{
	uint32_t vox[4];
};

struct VoxelRenderData
{
	uint32_t color;
};
#pragma pack(pop)

struct VoxelInfo
{
	uint32_t	sceneVoxCacheCount;
	double		sceneVoxCacheSize;

	uint32_t	sceneVoxOctreeCount;
	double		sceneVoxOctreeSize;

};

class ThesisSolution : public SolutionI
{
	private:
		SceneI*					currentScene;

		DeferredRenderer&		dRenderer;

		Shader					vertexDebugVoxel;
		Shader					fragmentDebugVoxel;

		Shader					vertexVoxelizeObject;
		Shader					geomVoxelizeObject;
		Shader					fragmentVoxelizeObject;
		Shader					computeVoxelizeCount;
		Shader					computePackObjectVoxels;
		Shader					computeDetermineVoxSpan;

		FrameTransformBuffer	cameraTransform;

		// Voxel Cache
		StructuredBuffer<ObjGridInfo>			objectGridInfo;
		StructuredBuffer<VoxelData>				voxelData;
		StructuredBuffer<VoxelRenderData>		voxelRenderData;
		StructuredBuffer<uint32_t>				voxelCacheUsageSize;
		VoxelDebugVAO							voxelVAO;

		// Relative Transform buffer for rigid movements (for scene's draw buffer)
		StructuredBuffer<ModelTransform>		relativeTransformBuffer;

		// GUI
		TwBar*									bar;
		double									frameTime;
		VoxelInfo								voxInfo;
		
		// Uncomment this for debugging voxelization 
		// Normally this texture allocated and deallocated 
		// at init time and
		// Comment the one at Init function
		//VoxelRenderTexture voxelRenderTexture;

		// Cuda Segment
		GICudaVoxelScene		voxelScene;

		static size_t			InitialObjectGridSize;
		static size_t			InitialVoxelBufferSizes;
		static size_t			MaxVoxelCacheSize;

	protected:
		void					CreateVoxel(size_t index);
		void					DrawVoxel(size_t index);

	public:
								ThesisSolution(DeferredRenderer&);
								ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&	operator=(const ThesisSolution&) = delete;
		virtual					~ThesisSolution() = default;

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Release() override;
		void					Frame(const Camera&) override;
		void					SetFPS(double fpsMS) override;
};
#endif //__THESISSOLUTION_H__