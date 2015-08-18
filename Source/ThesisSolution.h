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

struct VoxelGridInfoGL
{
	IEVector4		posSpan;
	uint32_t		dimension[4];
};
#pragma pack(pop)

struct VoxelInfo
{
	uint32_t	sceneVoxCacheCount;
	double		sceneVoxCacheSize;

	uint32_t	sceneVoxOctreeCount;
	double		sceneVoxOctreeSize;

};

enum ThesisRenderScheme
{
	GI_DEFERRED,
	GI_LIGHT_INTENSITY,
	GI_VOXEL_PAGE,
	GI_VOXEL_CACHE
};

class ThesisSolution : public SolutionI
{
	private:
		SceneI*					currentScene;

		DeferredRenderer&		dRenderer;

		Shader					vertexDebugVoxel;
		Shader					vertexDebugWorldVoxel;
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

		// Utility Buffers
		// Relative Transform buffer for rigid movements (for scene's draw buffer)
		StructuredBuffer<ModelTransform>		relativeTransformBuffer;
		StructuredBuffer<VoxelGridInfoGL>		gridInfoBuffer;

		// GUI
		TwBar*									bar;
		bool									giOn;
		VoxelInfo								voxInfo;
		double									frameTime;
		double									ioTime;
		double									transformTime;
		double									svoTime;
		double									debugVoxTransferTime;

		ThesisRenderScheme						renderScheme;
		static const TwEnumVal					renderSchemeVals[];
		TwType									renderType;

		// Debug Rendering
		void									DebugRenderVoxelCache(const Camera& camera);
		void									DebugRenderVoxelPage(const Camera& camera, 
																	 VoxelDebugVAO& pageVoxels, 
																	 const CVoxelGrid& voxGrid);

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
		
	public:
								ThesisSolution(DeferredRenderer&, const IEVector3& intialCamPos);
								ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&	operator=(const ThesisSolution&) = delete;
								~ThesisSolution();

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Release() override;
		void					Frame(const Camera&) override;
		void					SetFPS(double fpsMS) override;
};
#endif //__THESISSOLUTION_H__