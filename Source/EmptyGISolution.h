/**

Empty Solution
Just Renders the scene

*/

#ifndef __EMPTYGISOLUTION_H__
#define __EMPTYGISOLUTION_H__

#include "SolutionI.h"
#include "Shader.h"
#include "FrameTransformBuffer.h"

class EmptyGISolution : public SolutionI
{
	private:
		SceneI*					currentScene;
		Shader					vertexGBufferWrite;
		Shader					fragmentGBufferWrite;
		FrameTransformBuffer	cameraTransform;

	protected:
	public:
								EmptyGISolution();
								~EmptyGISolution() = default;
		
		bool					IsCurrentScene(SceneI&) override;
		void					Init(SceneI&) override;
		void					Frame(const Camera&) override;
};
#endif //__EMPTYGISOLUTION_H__