#include "Scene.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"
#include "Globals.h"

ConstantScene::ConstantScene(const std::vector<std::string>& rigidFileNames,
							 const std::vector<std::string>& skeletalFileNames,
							 const std::vector<Light>& lights)
	: rigidFileNames(rigidFileNames)
	, skeletalFileNames(skeletalFileNames)
	, lights(lights)
	, sceneLights()
	, rigidBatch()
	, skeletalBatch()
	, meshBatch({&rigidBatch, &skeletalBatch})
	, materialCount(0)
	, objectCount(0)
	, drawCallCount(0)
	, totalPolygons(0)
{
	for(const MeshBatchI* batch : meshBatch)
	{
		materialCount += batch->MaterialCount();
		objectCount += batch->ObjectCount();
		drawCallCount += batch->DrawCount();
		totalPolygons += batch->PolyCount();
	}
}

SceneLights& ConstantScene::getSceneLights()
{
	return sceneLights;
}

const std::vector<MeshBatchI*>& ConstantScene::getBatches()
{
	return meshBatch;
}

size_t ConstantScene::ObjectCount() const
{
	return objectCount;
}

size_t ConstantScene::PolyCount() const
{
	return totalPolygons;
}

size_t ConstantScene::MaterialCount() const
{
	return  materialCount;
}

size_t ConstantScene::DrawCount() const
{
	return drawCallCount;
}

void ConstantScene::Update(double elapsedS)
{
	for(MeshBatchI* batch : meshBatch)
		batch->Update(elapsedS);
}

void ConstantScene::Load()
{
	rigidBatch = MeshBatch(rigidMeshVertexDefinition, sizeof(VAO), rigidFileNames);
	skeletalBatch = MeshBatch(skeletalMeshVertexDefinition, sizeof(VAOSkel), skeletalFileNames);
	sceneLights = SceneLights(lights);
}

void ConstantScene::Release()
{
	rigidBatch = MeshBatch();
	skeletalBatch = MeshBatch();
	sceneLights = SceneLights();
}