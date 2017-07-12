/**

Solution implementtion

*/

#ifndef __THESISSOLUTION_H__
#define __THESISSOLUTION_H__

#include "SolutionI.h"
#include "ThesisBar.h"
#include "IndirectBar.h"
#include "LightBar.h"
#include "GIVoxelCache.h"
#include "GIVoxelPages.h"
#include "GISparseVoxelOctree.h"
#include "ConeTraceTexture.h"

class DeferredRenderer;
class WindowInput;

class ThesisSolution : public SolutionI
{
	public:
		const OctreeParameters		octreeParams;
		const std::string			name;

		static constexpr GLsizei	TraceWidth = 1280;
		static constexpr GLsizei	TraceHeight = 720;

	private:
		// Entire Voxel Cache one Per Batch
		GIVoxelCache				voxelCaches;
		GIVoxelPages				voxelPages;
		GISparseVoxelOctree			voxelOctree;

		// Texture that is used for tracing
		ConeTraceTexture			coneTex;

		// Timings
		double						frameTime;
		double						directTime;
		double						ioTime;
		double						transTime;
		double						svoReconTime;
		double						svoInjectTime;
		double						svoAverageTime;
		double						coneTraceTime;
		double						miscTime;

		// Render Type
		RenderScheme				scheme;

		DeferredRenderer&			dRenderer;
		SceneI*						currentScene;
		
		// On/Off Switches
		bool						giOn;
		bool						aoOn;
        bool                        injectOn;

		// Light Params
		bool						directLighting;
		bool						ambientLighting;
		IEVector3					ambientColor;

		// GUI
		LightBar					lightBar;
		ThesisBar					thesisBar;
		IndirectBar					indirectBar;

	protected:
		
	public:
		// Constructors & Destructor
									ThesisSolution(uint32_t denseLevel,
												   uint32_t denseLevelCount,
												   uint32_t cascadeCont,
												   uint32_t cascadeBaseLevel,
												   float baseSpan,
												   WindowInput&,
												   DeferredRenderer&,
												   const std::string& name);
									ThesisSolution(const ThesisSolution&) = delete;
		const ThesisSolution&		operator=(const ThesisSolution&) = delete;
									~ThesisSolution() = default;

		// Interface
		bool						IsCurrentScene(SceneI&) override;
		void						Load(SceneI&) override;
		void						Release() override;
		void						Frame(const Camera&) override;
		void						SetFPS(double fpsMS) override;

		const std::string&			Name() const override;

		// Key Callbacks
		void						Next();
		void						Previous();
		void						Up();
		void						Down();
};
#endif //__THESISSOLUTION_H__