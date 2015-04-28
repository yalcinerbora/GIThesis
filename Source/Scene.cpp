#include "Scene.h"
#include "Globals.h"
#include "GFGLoader.h"

const char* Scene::sponzaFileName = "crySponza.gfg";
const char* Scene::cornellboxFileName = "cornell.gfg";

Scene::Scene(const char* sceneFileName)
	: sceneVertex({element, 3})
	, drawParams()
{
	GFGLoader::LoadGFG(sceneVertex, drawParams, sceneFileName);
}

// Interface
void Scene::Draw()
{
	sceneVertex.Bind();
	drawParams.Draw();
}