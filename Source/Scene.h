#pragma once
/**

*/

#include "SceneI.h"
#include "MeshBatch.h"
#include "MeshBatchSkeletal.h"
#include "SceneLights.h"

// Constant Scene means that new objects cannot be added
// after initialization time (can be updated freely)
class ConstantScene : public SceneI
{
	// Consta Scene means that objects cannot be
	private:
		const std::vector<std::string>		rigidFileNames;
		const std::vector<std::string>		skeletalFileNames;
		const std::vector<Light>			lights;

		// GPU
		SceneLights							sceneLights;
		MeshBatch							rigidBatch;
		MeshBatchSkeletal					skeletalBatch;

		// Batch References
		std::vector<MeshBatchI*>			meshBatch;

		// Some Data Related to the scene
		size_t								materialCount;
		size_t								objectCount;
		size_t								drawCallCount;
		size_t								totalPolygons;

	protected:
	public:
		// Constructors & Destructor
											ConstantScene(const std::vector<std::string>& rigidFileNames,
														  const std::vector<std::string>& skeletalFileNames,
														  const std::vector<Light>& lights);
											ConstantScene(const ConstantScene&) = delete;
		ConstantScene&						operator=(const ConstantScene&) = delete;
											~ConstantScene() = default;
		
		const std::vector<MeshBatchI*>&		getBatches() override;
		SceneLights&						getSceneLights() override;

		size_t								ObjectCount() const override;
		size_t								PolyCount() const override;
		size_t								MaterialCount() const override;
		size_t								DrawCount() const override;

		void								Update(double elapsedS) override;
		void								Load() override;
		void								Release() override;
};