#pragma once

#include <vector>
#include <string>
#include "AntBar.h"
#include "RenderSelect.h"

class SceneLights;

class ThesisBar : public AntBar
{
	private:
		static constexpr char*		ThesisBarName = "ThesisGI";

		VoxelRenderSelect			renderSelect;

	protected:
	public:
		// Constructors & Destructor
									ThesisBar() = default;
									ThesisBar(const SceneLights& lights,
											  RenderScheme& scheme,
											  double& frameTime,
											  double& directTime,
											  double& ioTime,
											  double& transTime,
											  double& svoReconTime,
											  double& svoAverageTime,
											  double& coneTraceTime,
											  double& miscTime,
											  int totalCascades,
											  int minSVO, int maxSVO);
		ThesisBar&					operator=(ThesisBar&&) = default;
									~ThesisBar() = default;

		// Timing Related
		bool						DoTiming() const;
		int							Light() const;
		int							LightLevel() const;
		int							SVOLevel() const;
		int							CacheCascade() const;
		int							PageCascade() const;

		OctreeRenderType			SVORenderType() const;
		VoxelRenderType				CacheRenderType() const;
		VoxelRenderType				PageRenderType() const;

		void						Next();
		void						Previous();
		void						Up();
		void						Down();
};