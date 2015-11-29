#include "Scene.h"
#include "Globals.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"

const char* Scene::sponzaFileName = "crySponza.gfg";
const char* Scene::cornellboxFileName = "cornell.gfg";
const char* Scene::movingObjectsFileName = "movingObjects.gfg";

const uint32_t Scene::sponzaSVOLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	4096,
	15 * 1024,
	65 * 1024,
	300 * 1024,
	1300 * 1024,
	1000 * 1024,
	1200 * 1024
};

const uint32_t Scene::cornellSVOLevelSizes[] =
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

const uint32_t Scene::movingObjectsSVOLevelSizes[] =
{
	1,
	8,
	64,
	512,
	1024,
	1024,
	1024,
	1024,
	2048,
	8192,
	30 * 1024,
	120 * 1024,
};

const uint32_t Scene::sponzaSVOTotalSize = 5222 * 1024;
const uint32_t Scene::cornellSVOTotalSize = 4200 * 1024;
const uint32_t Scene::movingObjectsTotalSize = 62 * 1024;

static_assert(sizeof(Scene::cornellSVOLevelSizes) / sizeof(uint32_t) == 12, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::sponzaSVOLevelSizes) / sizeof(uint32_t) == 12, "Scene Size Ratio Mismatch");
static_assert(sizeof(Scene::movingObjectsSVOLevelSizes) / sizeof(uint32_t) == 12, "Scene Size Ratio Mismatch");

Scene::Scene(const char* sceneFileName,
			 const Array32<Light>& lights,
			 float minVoxSpan,
			 uint32_t totalSVOArraySize,
			 const uint32_t svoLevelSizes[])
	: sceneVertex({element, 3})
	, drawParams()
	, sceneLights(lights)
	, minSpan(minVoxSpan)
	, svoLevelSizes(svoLevelSizes)
	, svoTotalSize(totalSVOArraySize)
{
	IETimer timer;
	timer.Start();

	SceneParams sceneParams = {0};
	GFGLoader::LoadGFG(sceneParams, sceneVertex, drawParams, sceneFileName);
	drawParams.SendToGPU();
	timer.Stop();

	materialCount = sceneParams.materialCount;
	objectCount = sceneParams.objectCount;
	drawCallCount = sceneParams.drawCallCount;
	totalPolygons = sceneParams.totalPolygons;

	GI_LOG("Loading \"%s\" complete", sceneFileName, timer.ElapsedMilliS());
	GI_LOG("\tDuration : %f ms", timer.ElapsedMilliS());
	GI_LOG("\tMaterials : %d", materialCount);
	GI_LOG("\tMeshes : %d", objectCount);
	GI_LOG("\tDrawPoints : %d", drawCallCount);
	GI_LOG("\tPolyCount : %d", totalPolygons);
	GI_LOG("----------");
}

DrawBuffer& Scene::getDrawBuffer()
{
	return drawParams;
}

GPUBuffer& Scene::getGPUBuffer()
{
	return sceneVertex;
}

SceneLights& Scene::getSceneLights()
{
	return sceneLights;
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

float Scene::MinSpan() const
{
	return minSpan;
}

uint32_t Scene::SVOTotalSize() const
{
	return svoTotalSize;
}

const uint32_t* Scene::SVOLevelSizes() const
{
	return svoLevelSizes;
}