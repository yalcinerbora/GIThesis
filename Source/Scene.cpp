#include "Scene.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"

const uint32_t Scene::bigSizes[] =
{
	11,					// Root
	18,					// 1
	164,				// 2
	1512,				// 3
	11024,				// 4
	18192,				// 5
	16320 * 1024,		// 6
	16560 * 1024,		// 7
	17000 * 1024,		// 8
	17000 * 1024,		// 9
	17000 * 1024,		// 10
	17000 * 1024,		// 11
	15000 * 1024,		// 12
};

const uint32_t Scene::sponzaSceneLevelSizes[] =
{
	1,                  // Root
	8,                  // 1
	64,                 // 2
	512,                // 3
    8192,               // 4
	4 * 1024,           // 5
	24 * 1024,          // 6
	128 * 1024,         // 7
	400 * 1024,         // 8
	1800 * 1024,        // 9
	6500 * 1024,        // 10
	13000 * 1024,       // 11
	5500 * 1024         // 12
};

const uint32_t Scene::cornellSceneLevelSizes[] =
{
    1,                  // Root
    8,                  // 1
    64,                 // 2
    512,                // 3
    1024,               // 4
    8192,               // 5
    5 * 1024,           // 6
    25 * 1024,          // 7
    100 * 1024,         // 8
    300 * 1024,         // 9
    1000 * 1024,        // 10
    3500 * 1024,        // 11
    14000 * 1024        // 12
};

const uint32_t Scene::sibernikSceneLevelSizes[] =
{
    1,                  // Root
    8,                  // 1
    64,                 // 2
    512,                // 3
    1024,               // 4
    8192,               // 5
    3 * 1024,           // 6
    10 * 1024,          // 7
    100 * 1024,         // 8
    300 * 1024,         // 9
    1000 * 1024,        // 10
    3100 * 1024,        // 11
    0 * 1024            // 12
};

const uint32_t Scene::dynamicSceneLevelSizes[] =
{
    1,                  // Root
    8,                  // 1
    64,                 // 2
    512,                // 3
    1024,               // 4
    8192,               // 5
    10 * 1024,          // 6
    50 * 1024,          // 7
    120 * 1024,         // 8
    500 * 1024,         // 9
    1500 * 1024,        // 10
    6000 * 1024,        // 11
    15000 * 1024        // 12
};

static_assert(sizeof(Scene::sponzaSceneLevelSizes) / sizeof(uint32_t) == 13, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::cornellSceneLevelSizes) / sizeof(uint32_t) == 13, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::sibernikSceneLevelSizes) / sizeof(uint32_t) == 13, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::dynamicSceneLevelSizes) / sizeof(uint32_t) == 13, "Scene Size Ratio Mismatch");

Scene::Scene(const Array32<MeshBatchI*> batches,
			 const Array32<Light>& lights,
			 const uint32_t svoLevelSizes[])
	: sceneLights(lights)
	, svoLevelSizes(svoLevelSizes)
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