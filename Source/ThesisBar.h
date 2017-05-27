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

		//double						frameTime;

		//double						directTime;
		//double						ioTime;
		//double						transTime;
		//double						svoReconTime;
		//double						svoAverageTime;
		//double						coneTraceTime;

		//int							totalCascades;
		//int							minSVO;
		//int							maxSVO;
		//
		//std::vector<int>			svoVoxelCounts;
		//std::string					svoTotalSize;
		//std::vector<int>			pageCascadeCounts;
		//std::vector<std::string>	pageCascadeSize;
		//std::vector<int>			cacheCascadeCounts;
		//std::vector<std::string>	cacheCascadeSize;
		//
		//bool						giOn;
		//bool						aoOn;
		//bool						injectOn;

	protected:
	public:
		// Constructors & Destructor
									ThesisBar() = default;
									ThesisBar(const SceneLights& lights, 
											  RenderScheme& scheme,
											  int totalCascades,
											  int minSVO, int maxSVO);
									~ThesisBar() = default;

		// Timing Related
		int							CurrentLight() const;
		int							CurrentLevel() const;
		int							CurrentSVOLevel() const;
		int							CurrentCacheCascade() const;
		int							CurrentPageLevel() const;

		void						Next();
		void						Previous();
		void						Up();
		void						Down();

		//// Timing Related
		//void						SetTotalTime();
		//void						SetDirectTime();
		//void						SetVoxelIOTime();
		//void						SetVoxelTransformTime();
		//void						SetSVOReconTime();
		//void						SetSVOAverageTime();
		//void						SetMiscTime();


	
};