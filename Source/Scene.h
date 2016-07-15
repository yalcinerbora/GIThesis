/**

*/

#ifndef __SCENE_H__
#define __SCENE_H__

#include "SceneI.h"
#include "MeshBatchI.h"
#include "SceneLights.h"

class Scene : public SceneI
{
	private:
		// Props
		SceneLights					sceneLights;
		std::vector<MeshBatchI*>	meshBatch;

		// Some Data Related to the scene
		size_t						materialCount;
		size_t						objectCount;
		size_t						drawCallCount;
		size_t						totalPolygons;

		uint32_t					svoTotalSize;
		const uint32_t*				svoLevelSizes;

	protected:
	public:
		// Constructors & Destructor
								Scene(const Array32<MeshBatchI*> batches,
									  const Array32<Light>& light,
									  uint32_t totalSVOArraySize,
									  const uint32_t svoLevelSizes[]);
								~Scene() = default;

		static const uint32_t	sponzaSceneLevelSizes[];
		static const uint32_t	cornellSceneLevelSizes[];
		static const uint32_t	cubeSceneLevelSizes[];
		static const uint32_t	sibernikSceneLevelSizes[];
		static const uint32_t	tinmanSceneLevelSizes[];

		static const uint32_t	tinmanSceneTotalSize;
		static const uint32_t	sponzaSceneTotalSize;
		static const uint32_t	cornellSceneTotalSize;
		static const uint32_t	cubeSceneTotalSize;
		static const uint32_t	sibernikSceneTotalSize;
		
		Array32<MeshBatchI*>	getBatches() override;
		SceneLights&			getSceneLights() override;

		size_t					ObjectCount() const override;
		size_t					PolyCount() const override;
		size_t					MaterialCount() const override;
		size_t					DrawCount() const override;

		void					Update(double elapsedS) override;

		uint32_t				SVOTotalSize() const override;
		const uint32_t*			SVOLevelSizes() const override;
};

#endif //__SCENE_H__