/**

*/

#ifndef __SCENE_H__
#define __SCENE_H__

#include "SceneI.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"
#include "SceneLights.h"

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
		GPUBuffer				sceneVertex;
		DrawBuffer				drawParams;

		SceneLights				sceneLights;

		// Some Data Related to the scene
		size_t					materialCount;
		size_t					objectCount;
		size_t					drawCallCount;
		size_t					totalPolygons;

	protected:
	public:
		// Constructors & Destructor
								Scene(const char* sceneFileName,
									  const Array32<Light>& light);
								~Scene() = default;

		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;

		DrawBuffer&				getDrawBuffer() override;
		GPUBuffer&				getGPUBuffer() override;

		size_t					ObjectCount() const override;
		size_t					PolyCount() const override;
		size_t					MaterialCount() const override;
		size_t					DrawCount() const override;
};

#endif //__SCENE_H__