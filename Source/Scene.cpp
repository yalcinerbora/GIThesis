#include "Scene.h"
#include "Globals.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"

const char* Scene::sponzaFileName = "crySponza.gfg";
const char* Scene::cornellboxFileName = "cornell.gfg";
//const char* Scene::movingObjectsFileName = "movingObjects.gfg";
const char* Scene::movingObjectsFileName = "mvObj1.gfg";

Scene::Scene(const char* sceneFileName,
			 const Array32<Light>& lights,
			 float minSpan,
			 float svoMultiplier)
	: sceneVertex({element, 3})
	, drawParams()
	, sceneLights(lights)
	, minSpan(minSpan)
	, svoMultiplier(svoMultiplier)
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

float Scene::SVOMultiplier() const
{
	return svoMultiplier;
}