/**

*/

#ifndef __SCENE_H__
#define __SCENE_H__

#include "SceneI.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

struct SceneParams
{
	size_t				materialCount;
	size_t				objectCount;
	size_t				drawCallCount;
	size_t				totalPolygons;
};


class Scene : public SceneI
{
	private:
		// Props
		GPUBuffer			sceneVertex;
		DrawBuffer			drawParams;
	//	AABBBuffer			aabbBuffer;

		// Some Data Related to the scene
		size_t				materialCount;
		size_t				objectCount;
		size_t				drawCallCount;
		size_t				totalPolygons;

	protected:
	public:
		// Constructors & Destructor
								Scene(const char* sceneFileName);
								~Scene() = default;

		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;

		void					Draw() override;
		DrawBuffer&				getDrawBuffer() override;
		size_t					ObjectCount() const override;
};

#endif //__SCENE_H__