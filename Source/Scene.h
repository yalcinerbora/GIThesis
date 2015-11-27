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

		float					minSpan;
		float					svoMultiplier;

	protected:
	public:
		// Constructors & Destructor
								Scene(const char* sceneFileName,
									  const Array32<Light>& light,
									  float minSpan,
									  float svoMultipler);
								~Scene() = default;

		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;
		static const char*		movingObjectsFileName;

		static const uint32_t	sponzaSVOLevelSizes[];
		static const uint32_t	cornellSVOLevelSizes[];
		static const uint32_t	movingObjectsSVOLevelSizes[];
		

		DrawBuffer&				getDrawBuffer() override;
		GPUBuffer&				getGPUBuffer() override;
		SceneLights&			getSceneLights() override;

		size_t					ObjectCount() const override;
		size_t					PolyCount() const override;
		size_t					MaterialCount() const override;
		size_t					DrawCount() const override;

		float					MinSpan() const override;
		float					SVOMultiplier() const override;
};

#endif //__SCENE_H__