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
		uint32_t				svoTotalSize;
		const uint32_t*			svoLevelSizes;

	protected:
	public:
		// Constructors & Destructor
								Scene(const char* sceneFileName,
									  const Array32<Light>& light,
									  float minVoxSpan,
									  uint32_t totalSVOArraySize,
									  const uint32_t svoLevelSizes[]);
								~Scene() = default;

		// Static Files
		static const char*		sponzaFileName;
		static const char*		cornellboxFileName;
		static const char*		movingObjectsFileName;

		static const uint32_t	sponzaSVOLevelSizes[];
		static const uint32_t	cornellSVOLevelSizes[];
		static const uint32_t	movingObjectsSVOLevelSizes[];

		static const uint32_t	sponzaSVOTotalSize;
		static const uint32_t	cornellSVOTotalSize;
		static const uint32_t	movingObjectsTotalSize;
		

		DrawBuffer&				getDrawBuffer() override;
		GPUBuffer&				getGPUBuffer() override;
		SceneLights&			getSceneLights() override;

		size_t					ObjectCount() const override;
		size_t					PolyCount() const override;
		size_t					MaterialCount() const override;
		size_t					DrawCount() const override;

		float					MinSpan() const override;
		uint32_t				SVOTotalSize() const override;
		const uint32_t*			SVOLevelSizes() const override;
};

#endif //__SCENE_H__