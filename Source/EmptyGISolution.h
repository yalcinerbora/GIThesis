/**

Empty Solution
Just Renders the scene

*/

#ifndef __EMPTYGISOLUTION_H__
#define __EMPTYGISOLUTION_H__

#include "SolutionI.h"

class DeferredRenderer;

class EmptyGISolution : public SolutionI
{
	private:
		SceneI*					currentScene;
		DeferredRenderer&		dRenderer;

	protected:
	public:
								EmptyGISolution(DeferredRenderer&);
								~EmptyGISolution() = default;
		
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Frame(const Camera&) override;
};
#endif //__EMPTYGISOLUTION_H__