#include "Scene.h"
#include "Globals.h"
#include "GFGLoader.h"
#include "IEUtility/IETimer.h"
#include "Macros.h"

const char* Scene::sponzaFileName = "crySponza.gfg";
const char* Scene::cornellboxFileName = "cornell.gfg";

Scene::Scene(const char* sceneFileName)
	: sceneVertex({element, 3})
	, drawParams()
{
	IETimer timer;
	timer.Start();

	SceneParams sceneParams = {0};
	GFGLoader::LoadGFG(sceneParams, sceneVertex, drawParams, sceneFileName);
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

// Interface
void Scene::Draw()
{
	sceneVertex.Bind();
	drawParams.Draw();
}