/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include "SolutionI.h"
#include "ThesisBar.h"
#include "IndirectBar.h"
#include "LightBar.h"
#include "VoxelCacheBatches.h"

class DeferredRenderer;
class WindowInput;

class ThesisSolution : public SolutionI
{
	private:
		// Entire Voxel Cache one Per Batch
		VoxelCacheBatches				voxelCaches;
		//GIVoxelPages					voxelPages;

		//// Voxel Render Shaders
		//Shader								vertexDebugVoxel;
		//Shader								vertexDebugVoxelSkeletal;
		//Shader								vertexDebugWorldVoxel;
		//Shader								fragmentDebugVoxel;

		//// Voxel Cache for each cascade
		//std::vector<SceneVoxCache>			voxelCaches;

		//// Cuda Stuff
		//std::vector<GICudaVoxelScene>			voxelScenes;
		//GISparseVoxelOctree					voxelOctree;

		//// Utility(Debug) Buffers (Page Voxel Rendering)
		//StructuredBuffer<VoxelGridInfoGL>	gridInfoBuffer;
		//StructuredBuffer<VoxelNormPos>		voxelNormPosBuffer;
		//StructuredBuffer<uchar4>			voxelColorBuffer;

		//cudaGraphicsResource_t				vaoNormPosResource;
		//cudaGraphicsResource_t				vaoRenderResource;

		const std::string					name;

		DeferredRenderer&					dRenderer;
		SceneI*								currentScene;
		
		// On/Off Switches
		bool								giOn;
		bool								aoOn;
        bool                                injectOn;

		// Light Params
		bool								directLighting;
		bool								ambientLighting;
		IEVector3							ambientColor;

		// Times
		double								directTime;
		double								ioTime;
		double								transTime;
		double								svoReconTime;
		double								svoAverageTime;
		double								coneTraceTime;
		double								miscTime;

		// Render Type
		RenderScheme						scheme;
											
		// GUI
		LightBar							lightBar;
		ThesisBar							thesisBar;
		IndirectBar							indirectBar;

		const uint32_t						cascadeCount;


		//// Debug Rendering					
		//double								DebugRenderVoxelCache(const Camera& camera,
		//														  SceneVoxCache&);
		//void								DebugRenderVoxelPage(const Camera& camera,
		//														 VoxelDebugVAO& pageVoxels,
		//														 const CVoxelGrid& voxGrid,
		//														 uint32_t offset,
		//														 uint32_t voxCount);
		//double								DebugRenderSVO(const Camera& camera);

		 // Voxelizes the scene for a cache level
		//double								LoadBatchVoxels(MeshBatchI* batch);
		//bool								LoadVoxel(std::vector<SceneVoxCache>& scenes,
		//											  const char* gfgFileName, uint32_t cascadeCount,
		//											  bool isSkeletal,
  //                                                    int repeatCount = 1);
		//void								LinkCacheWithVoxScene(GICudaVoxelScene&, 
		//														  SceneVoxCache&,
		//														  float coverageRatio);
													 
		//// Cuda Segment
		//static const size_t		InitialObjectGridSize;

		//AOBar					aoBar;
		//unsigned int			svoRenderLevel;
		//unsigned int			traceType;

	protected:
		
	public:
		// Constructors & Destructor
											ThesisSolution(uint32_t cascadeCount,
														   WindowInput&,
														   DeferredRenderer&,
														   const std::string& name);
											ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&				operator=(const ThesisSolution&) = delete;
											~ThesisSolution() = default;

		// Interface
		bool								IsCurrentScene(SceneI&) override;
		void								Load(SceneI&) override;
		void								Release() override;
		void								Frame(const Camera&) override;
		void								SetFPS(double fpsMS) override;

		// Key Callbacks
		void								Next();
		void								Previous();
		void								Up();
		void								Down();
};
#endif //__THESISSOLUTION_H__