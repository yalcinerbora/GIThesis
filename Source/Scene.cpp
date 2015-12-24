#include "Scene.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"

//const uint32_t Scene::sponzaSceneLevelSizes[] =
//{
//	1,
//	8,
//	64,
//	512,
//	1024,
//	4096,
//	15 * 1024,
//	80 * 1024,
//	300 * 1024,
//	1400 * 1024,
//	1100 * 1024,
//	1200 * 1024
//};

const uint32_t Scene::sponzaSceneLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	8192,
	45 * 1024,
	280 * 1024,
	600 * 1024,
	2400 * 1024,
	2100 * 1024,
	2200 * 1024
};

const uint32_t Scene::cornellSceneLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	1024,
	2048,
	10 * 1024,
	35 * 1024,
	190 * 1024,
	750 * 1024,
	2200 * 1024
};

const uint32_t Scene::cubeSceneLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	1024,
	1024,
	4 * 1024,
	8 * 1024,
	25 * 1024,
	80 * 1024,
	300 * 1024,
};

const uint32_t Scene::sponzaSceneTotalSize = 8501 * 1024;
const uint32_t Scene::cornellSceneTotalSize = 3190 * 1024;
const uint32_t Scene::cubeSceneTotalSize = 421 * 1024;

static_assert(sizeof(Scene::sponzaSceneLevelSizes) / sizeof(uint32_t) == 12, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::cornellSceneLevelSizes) / sizeof(uint32_t) == 12, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::cubeSceneLevelSizes) / sizeof(uint32_t) == 12, "Scene Size Ratio Mismatch");

Scene::Scene(const Array32<MeshBatchI*> batches,
			 const Array32<Light>& lights,
			 uint32_t totalSVOArraySize,
			 const uint32_t svoLevelSizes[])
	: sceneLights(lights)
	, svoLevelSizes(svoLevelSizes)
	, svoTotalSize(totalSVOArraySize)
	, meshBatch(batches.arr, batches.arr + batches.length)
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

SceneLights& Scene::getSceneLights()
{
	return sceneLights;
}

Array32<MeshBatchI*> Scene::getBatches()
{
	return Array32<MeshBatchI*>{meshBatch.data(), static_cast<uint32_t>(meshBatch.size())};
}

uint32_t Scene::SVOTotalSize() const
{
	return svoTotalSize;
}

const uint32_t* Scene::SVOLevelSizes() const
{
	return svoLevelSizes;
}

size_t Scene::ObjectCount() const
{
	return objectCount;
}

size_t Scene::PolyCount() const
{
	return totalPolygons;
}

size_t Scene::MaterialCount() const
{
	return  materialCount;
}

size_t Scene::DrawCount() const
{
	return drawCallCount;
}

void Scene::Update(double elapsedS)
{
	for(MeshBatchI* batch : meshBatch)
		batch->Update(elapsedS);
}