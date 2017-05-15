#include "Scene.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"
#include "Globals.h"

ConstantScene::ConstantScene(const std::string& name,
							 const std::vector<std::string>& rigidFileNames,
							 const std::vector<std::string>& skeletalFileNames,
							 const std::vector<Light>& lights)
	: name(name)
	, rigidFileNames(rigidFileNames)
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
{}

//MeshBatch& ConstantScene::getRigidBatch()
//{
//	return rigidBatch;
//}
//
//MeshBatchSkeletal& ConstantScene::getSkeletalBatch()
//{
//	return skeletalBatch;
//}

SceneLights& ConstantScene::getSceneLights()
{
	return sceneLights;
}

const SceneLights& ConstantScene::getSceneLights() const
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

void ConstantScene::Initialize()
{}

void ConstantScene::Update(double elapsedS)
{
	for(MeshBatchI* batch : meshBatch)
		batch->Update(elapsedS);
}

void ConstantScene::Load()
{
	rigidBatch = MeshBatch(rigidMeshVertexDefinition, sizeof(VAO), rigidFileNames);
	skeletalBatch = MeshBatchSkeletal(skeletalMeshVertexDefinition, sizeof(VAOSkel), skeletalFileNames);
	sceneLights = SceneLights(lights);

	for(const MeshBatchI* batch : meshBatch)
	{
		materialCount += batch->MaterialCount();
		objectCount += batch->ObjectCount();
		drawCallCount += batch->DrawCount();
		totalPolygons += batch->PolyCount();
	}

	Initialize();
}

void ConstantScene::Release()
{
	rigidBatch = MeshBatch();
	skeletalBatch = MeshBatchSkeletal();
	sceneLights = SceneLights();
}

const std::string& ConstantScene::Name() const
{
	return name;
}