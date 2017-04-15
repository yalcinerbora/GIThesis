/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include <vector>
#include <list>
#include <AntTweakBar.h>
#include "EmptyGISolution.h"
#include "Shader.h"
#include "FrameTransformBuffer.h"
#include "GICudaVoxelScene.h"
#include "StructuredBuffer.h"
#include "IEUtility/IEVector3.h"
#include "VoxelRenderTexture.h"
#include "VoxelDebugVAO.h"
#include "DrawBuffer.h"
#include "MeshBatchI.h"
#include "GISparseVoxelOctree.h"
#include "VoxelSceneCache.h"

class DeferredRenderer;

enum ThesisRenderScheme
{
	GI_DEFERRED,
	GI_LIGHT_INTENSITY,
	GI_SVO_DEFERRED,
	GI_SVO_LEVELS,
	GI_VOXEL_PAGE,
	GI_VOXEL_CACHE,
	GI_END,
};

class AOBar
{
	public:
		TwBar*	bar;
		float	angleDegree;
		float	sampleFactor;
		float	maxDistance;
		float	falloffFactor;
		float	intensityAO;
		float	intensityGI;
		bool	hidden;
		bool	specular;

				AOBar();
				~AOBar();

	void HideBar(bool);
};

class ThesisSolution : public EmptyGISolution
{
	private:
		// Voxel Render Shaders
		Shader								vertexDebugVoxel;
		Shader								vertexDebugVoxelSkeletal;
		Shader								vertexDebugWorldVoxel;
		Shader								fragmentDebugVoxel;

		FrameTransformBuffer				cameraTransform;

		// Voxel Cache for each cascade
		std::vector<SceneVoxCache>			voxelCaches;

		// Cuda Stuff
		std::vector<GICudaVoxelScene>		voxelScenes;
		GISparseVoxelOctree					voxelOctree;

		// Utility(Debug) Buffers (Page Voxel Rendering)
		StructuredBuffer<VoxelGridInfoGL>	gridInfoBuffer;
		StructuredBuffer<VoxelNormPos>		voxelNormPosBuffer;
		StructuredBuffer<uchar4>			voxelColorBuffer;

		cudaGraphicsResource_t				vaoNormPosResource;
		cudaGraphicsResource_t				vaoRenderResource;

		// GUI								
		TwBar*								bar;
		bool								giOn;
		bool								aoOn;
        bool                                injectOn;

		// Times
		double								ioTime;
		double								transformTime;
		double								svoReconTime;
		double								svoInjectTime;
		double								svoAvgTime;
		double								giTime;
		double								miscTime;
		double								totalTime;
											
		ThesisRenderScheme					renderScheme;
		static const TwEnumVal				renderSchemeVals[];
		TwType								renderType;
											
		// Debug Rendering					
		double								DebugRenderVoxelCache(const Camera& camera,
																  SceneVoxCache&);
		void								DebugRenderVoxelPage(const Camera& camera,
																 VoxelDebugVAO& pageVoxels,
																 const CVoxelGrid& voxGrid,
																 uint32_t offset,
																 uint32_t voxCount);
		double								DebugRenderSVO(const Camera& camera);

		 // Voxelizes the scene for a cache level
		double								LoadBatchVoxels(MeshBatchI* batch);
		bool								LoadVoxel(std::vector<SceneVoxCache>& scenes,
													  const char* gfgFileName, uint32_t cascadeCount,
													  bool isSkeletal,
                                                      int repeatCount = 1);
		void								LinkCacheWithVoxScene(GICudaVoxelScene&, 
																  SceneVoxCache&,
																  float coverageRatio);
													 
		// Cuda Segment
		static const size_t		InitialObjectGridSize;

		AOBar					aoBar;
		unsigned int			svoRenderLevel;
		unsigned int			traceType;

	protected:
		
	public:
								ThesisSolution(const std::string& name,
											   DeferredRenderer&, const IEVector3& intialCamPos);
								ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&	operator=(const ThesisSolution&) = delete;
								~ThesisSolution();

		// Globals
		static const float		CascadeSpan;
		static const uint32_t	CascadeDim;

		// Interface
		bool					IsCurrentScene(SceneI&) override;
		void					Load(SceneI&) override;
		void					Release() override;
		void					Frame(const Camera&) override;

		// Input Calling Functions
		void					SVOLevelIncrement();
		void					SVOLevelDecrement();
		void					TraceTypeIncrement();
		void					TraceTypeDecrement();
};
#endif //__THESISSOLUTION_H__