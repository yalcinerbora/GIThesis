#pragma once
/**

*/

#include "SceneI.h"
#include "MeshBatch.h"
#include "MeshBatchSkeletal.h"
#include "SceneLights.h"

// Constant Scene means that new objects cannot be added
// after initialization time (can be updated moved etc..)
class ConstantScene : public SceneI
{
	// Consta Scene means that objects cannot be
	private:

	protected:
		// GPU
		SceneLights							sceneLights;
		MeshBatch							rigidBatch;
		MeshBatchSkeletal					skeletalBatch;

		const std::string					name;
		const std::vector<std::string>		rigidFileNames;
		const std::vector<std::string>		skeletalFileNames;
		const std::vector<Light>			lights;

		// Batch References
		std::vector<MeshBatchI*>			meshBatch;

		// Some Data Related to the scene
		size_t								materialCount;
		size_t								objectCount;
		size_t								drawCallCount;
		size_t								totalPolygons;

	public:
		// Constructors & Destructor
											ConstantScene(const std::string& name,
														  const std::vector<std::string>& rigidFileNames,
														  const std::vector<std::string>& skeletalFileNames,
														  const std::vector<Light>& lights);
											ConstantScene(const ConstantScene&) = delete;
		ConstantScene&						operator=(const ConstantScene&) = delete;
											~ConstantScene() = default;
		
		// Interface
		const std::vector<std::string>&		getBatchFileNames(uint32_t batchId) override;
		const std::vector<MeshBatchI*>&		getBatches() override;
		SceneLights&						getSceneLights() override;
		const SceneLights&					getSceneLights() const override;

		size_t								ObjectCount() const override;
		size_t								PolyCount() const override;
		size_t								MaterialCount() const override;
		size_t								DrawCount() const override;

		void								Initialize() override;
		void								Update(double elapsedS) override;
		void								Load() override;
		void								Release() override;

		const std::string&					Name() const override;
};