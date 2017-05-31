#pragma once

#include "AntBar.h"
#include "RenderSelect.h"

class SceneLights;

class EmptyGIBar : public AntBar
{
	private:
		static constexpr char*		EmptyGIBarName = "EmptyGI";

		RenderSelect				renderSelect;

	protected:
	public:
		// Constructors & Destructor
									EmptyGIBar() = default;
									EmptyGIBar(const SceneLights& lights,
											   RenderScheme& scheme,
											   double& frameTime,
											   double& shadowTime,
											   double& dPassTime,
											   double& gPassTime,
											   double& lightTime,
											   double& mergeTime);
		EmptyGIBar&					operator=(EmptyGIBar&&) = default;
									~EmptyGIBar() = default;

		// Timing Related
		bool						DoTiming() const;
		int							Light() const;
		int							LightLevel() const;

		void						Next();
		void						Previous();
		void						Up();
		void						Down();
};