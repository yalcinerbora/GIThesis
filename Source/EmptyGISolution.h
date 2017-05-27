/**

Empty Solution
Just Renders the scene

*/

#ifndef __EMPTYGISOLUTION_H__
#define __EMPTYGISOLUTION_H__

#include "SolutionI.h"
#include <cstdint>
#include <vector>
#include "IEUtility/IEVector3.h"
#include "LightBar.h"
#include "EmptyGIBar.h"

class DeferredRenderer;
class WindowInput;

class EmptyGISolution : public SolutionI
{
	private:
		const std::string					name;

		DeferredRenderer&					dRenderer;
		SceneI*								currentScene;
		
		// Light Params
		bool								directLighting;
		bool								ambientLighting;
		IEVector3							ambientColor;

		// Timing Params
		double								frameTime;
		double								shadowTime;
		double								dPassTime;
		double								gPassTime;
		double								lPassTime;
		double								mergeTime;

		// Render Type
		RenderScheme						scheme;

		// GUI
		LightBar							lightBar;
		EmptyGIBar							emptyGIBar;
	
	protected:
	public:
											EmptyGISolution(WindowInput&,
															DeferredRenderer&,
															const std::string& name);
											EmptyGISolution(const EmptyGISolution&) = delete;
		EmptyGISolution&					operator=(const EmptyGISolution&) = delete;
											~EmptyGISolution() = default;
	
		bool								IsCurrentScene(SceneI&) override;
		void								Load(SceneI&) override;
		void								Release() override;
		void								Frame(const Camera&) override;
		void								SetFPS(double fpsMS) override;

		const std::string&					Name() const override;

		// Key Callbacks
		void								Next();
		void								Previous();
		void								Up();
		void								Down();
};
#endif //__EMPTYGISOLUTION_H__