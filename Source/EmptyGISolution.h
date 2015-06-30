/**

Empty Solution
Just Renders the scene

*/

#ifndef __EMPTYGISOLUTION_H__
#define __EMPTYGISOLUTION_H__

#include "SolutionI.h"
#include <AntTweakBar.h>


class DeferredRenderer;

class EmptyGISolution : public SolutionI
{
	private:
		SceneI*					currentScene;
		DeferredRenderer&		dRenderer;

		TwBar*					bar;
		double					frameTime;

	protected:
	public:
								EmptyGISolution(DeferredRenderer&);
								~EmptyGISolution() = default;
		
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Release() override;
		void					Frame(const Camera&) override;
		void					SetFPS(double fpsMS) override;
};
#endif //__EMPTYGISOLUTION_H__