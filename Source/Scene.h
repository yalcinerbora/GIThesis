/**

*/

#ifndef __SCENE_H__
#define __SCENE_H__

#include "SceneI.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

class Scene : public SceneI
{
	private:
		// Props
		GPUBuffer			sceneVertex;
		DrawBuffer			drawParams;

	protected:
	public:
		// Constructors & Destructor
								Scene(const char* sceneFileName);
								~Scene() = default;

		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;

		void					Draw() override;
};

#endif //__SCENE_H__