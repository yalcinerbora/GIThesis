#include "Scene.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"

const uint32_t Scene::bigSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	8192,
	5320 * 1024,
	1560 * 1024,
	12000 * 1024,
	12000 * 1024,
	12000 * 1024,
	12000 * 1024,
	12000 * 1024,
};

const uint32_t Scene::sponzaSceneLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	4 * 1024,
	20 * 1024,
	62 * 1024,
	300 * 1024,
	1400 * 1024,
	4800 * 1024,
	7400 * 1024
};

const uint32_t Scene::cornellSceneLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	1024,
	4096,
	20 * 1024,
	50 * 1024,
	300 * 1024,
	1000 * 1024,
	4500 * 1024
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
	3 * 1024,
	8 * 1024,
	50 * 1024,
	200 * 1024,
	800 * 1024,
};

const uint32_t Scene::sibernikSceneLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	8192,
	45 * 1024,
	120 * 1024,
	400 * 1024,
	2000 * 1024,
	3800 * 1024,
	4000 * 1024
};

const uint32_t Scene::tinmanSceneLevelSizes[]
{
	1,
	8,
	64,
	512,
	1024,
	1024,
	1024,
	1024,
	1024,
	5 * 1024,
	10 * 1024,
	50 * 1024
};

const uint32_t Scene::bigTotalSize = 65 * 1024 * 1024;
const uint32_t Scene::sponzaSceneTotalSize = 15000 * 1024;
const uint32_t Scene::cornellSceneTotalSize = 5877 * 1024;
const uint32_t Scene::cubeSceneTotalSize = 1064 * 1024;
const uint32_t Scene::sibernikSceneTotalSize = 10400 * 1024;
const uint32_t Scene::tinmanSceneTotalSize = 71 * 1024;

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