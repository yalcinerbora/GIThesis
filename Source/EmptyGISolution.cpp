#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DeferredRenderer.h"
#include "Macros.h"
#include "SceneLights.h"
#include "Globals.h"

std::vector<TwLightCallbackLookup> EmptyGISolution::twCallbackLookup;

void TW_CALL EmptyGISolution::GetLightType(void *value, void *clientData)
{
	static const char* names[] =
	{
		"POINT",
		"DIRECTIONAL",
		"AREA",
	};
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	LightType t = lookup->solution->currentScene->getSceneLights().GetLightType(lookup->lightID);
	*static_cast<const char**>(value) = names[static_cast<int>(t)];
}

void TW_CALL EmptyGISolution::GetLightShadow(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	*static_cast<bool*>(value) = lookup->solution->currentScene->getSceneLights().GetLightShadow(lookup->lightID);
}

void TW_CALL EmptyGISolution::SetLightShadow(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->solution->currentScene->getSceneLights().ChangeLightShadow(lookup->lightID,
																	   *(static_cast<const bool*>(value)));
}

void TW_CALL EmptyGISolution::GetLightColor(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 color = lookup->solution->currentScene->getSceneLights().GetLightColor(lookup->lightID);
	*static_cast<IEVector3*>(value) = color;
}

void TW_CALL EmptyGISolution::SetLightColor(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->solution->currentScene->getSceneLights().ChangeLightColor(lookup->lightID,
																	  (*static_cast<const IEVector3*>(value)));
}
	    
void TW_CALL EmptyGISolution::GetLightIntensity(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	float intensity = lookup->solution->currentScene->getSceneLights().GetLightIntensity(lookup->lightID);
	*static_cast<float*>(value) = intensity;
}

void TW_CALL EmptyGISolution::SetLightIntensity(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->solution->currentScene->getSceneLights().ChangeLightIntensity(lookup->lightID,
																		  (*static_cast<const float*>(value)));
}
	    
void TW_CALL EmptyGISolution::GetLightPos(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 pos = lookup->solution->currentScene->getSceneLights().GetLightPos(lookup->lightID);
	*static_cast<IEVector3*>(value) = pos;
}

void TW_CALL EmptyGISolution::SetLightPos(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->solution->currentScene->getSceneLights().ChangeLightPos(lookup->lightID,
																	  (*static_cast<const IEVector3*>(value)));
}

void TW_CALL EmptyGISolution::GetLightDirection(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 intensity = lookup->solution->currentScene->getSceneLights().GetLightDir(lookup->lightID);
	*static_cast<IEVector3*>(value) = intensity.Normalize();
}

void TW_CALL EmptyGISolution::SetLightDirection(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->solution->currentScene->getSceneLights().ChangeLightDir(lookup->lightID,
																	(*static_cast<const IEVector3*>(value)));
}

void TW_CALL EmptyGISolution::GetLightRadius(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	float radius = lookup->solution->currentScene->getSceneLights().GetLightRadius(lookup->lightID);
	*static_cast<float*>(value) = radius;
}

void TW_CALL EmptyGISolution::SetLightRadius(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->solution->currentScene->getSceneLights().ChangeLightRadius(lookup->lightID,
																	(*static_cast<const float*>(value)));
}

EmptyGISolution::EmptyGISolution(DeferredRenderer& defferedRenderer)
	: currentScene(nullptr)
	, dRenderer(defferedRenderer)
	, bar(nullptr)
	, directLighting(true)
	, ambientLighting(true)
	, ambientColor(0.33f, 0.33f, 0.33f)
{}

bool EmptyGISolution::IsCurrentScene(SceneI& scene)
{
	return &scene == currentScene;
}
void EmptyGISolution::Init(SceneI& s)
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	currentScene = &s;

	// Bar Creation
	bar = TwNewBar("Ligths");
	TwDefine(" Ligths refresh=0.01 ");

	// FPS Show
	TwAddVarRO(bar, "fTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame(ms)' help='Frame Time in milliseconds..' ");
	TwAddVarRW(bar, "directLightOn", TW_TYPE_BOOLCPP,
			   &directLighting,
			   " label='Direct Light' help='Direct Ligting On Off' ");
	TwAddVarRW(bar, "ambientOn", TW_TYPE_BOOLCPP,
			   &ambientLighting,
			   " label='Ambient On' help='Ambient Ligting On Off' ");
	TwAddVarRW(bar, "aColor", TW_TYPE_COLOR3F,
			   &ambientColor,
			   " label='Ambient Color' help='Ambient Color'");

	TwAddSeparator(bar, NULL, NULL);
		
	std::string name;
	std::string params;
	twCallbackLookup.reserve(s.getSceneLights().Count());
	twCallbackLookup.clear();
	for(unsigned int i = 0; i < s.getSceneLights().Count(); i++)
	{
		twCallbackLookup.push_back({ i, this });
		LightType lightType = s.getSceneLights().GetLightType(i);

		name = "lType" + std::to_string(i);
		params = " label='Type' group='Light#"+ std::to_string(i);
		params += "' help='Light Type' ";
		TwAddVarCB(bar, name.c_str(), TW_TYPE_CDSTRING,
				   NULL,
				   GetLightType,
				   &(twCallbackLookup.back()),
				   params.c_str());

		name = "lShadow" + std::to_string(i);
		params = " label='Shadow' group='Light#" + std::to_string(i);
		params += "' help='Shadow Cast On/Off' ";
		TwAddVarCB(bar, name.c_str(), TW_TYPE_BOOLCPP,
				   SetLightShadow,
				   GetLightShadow,
				   &(twCallbackLookup.back()),
				   params.c_str());

		name = "lColor" + std::to_string(i);
		params = " label='Color' group='Light#" + std::to_string(i);
		params += "' help='Light Color' ";
		TwAddVarCB(bar, name.c_str(), TW_TYPE_COLOR3F,
				   SetLightColor,
				   GetLightColor,
				   &(twCallbackLookup.back()),
				   params.c_str());

		name = "lIntensity" + std::to_string(i);
		params = " label='Intensity' group='Light#" + std::to_string(i);
		params += "' help='Light Intensity' ";
		if(lightType == LightType::DIRECTIONAL)
		{
			params += " min=0.0 max=10.0 step=0.01 ";
		}
		else
		{
			params += "min=0.0 max=11000.0 step=10 ";
		}
		TwAddVarCB(bar, name.c_str(), TW_TYPE_FLOAT,
				   SetLightIntensity,
				   GetLightIntensity,
				   &(twCallbackLookup.back()),
				   params.c_str());

		if(lightType == LightType::DIRECTIONAL ||
		   lightType == LightType::AREA)
		{
			name = "lDirection" + std::to_string(i);
			params = " label='Direction' group='Light#" + std::to_string(i);
			params += "' help='Light Direction' ";
			if(i == 0) params += " opened=true ";
			TwAddVarCB(bar, name.c_str(), TW_TYPE_DIR3F,
					   SetLightDirection,
					   GetLightDirection,
					   &(twCallbackLookup.back()),
					   params.c_str());
		}

		if(lightType == LightType::POINT ||
		   lightType == LightType::AREA)
		{
			name = "lPosition" + std::to_string(i);
			params = " label='Position' group='Light#" + std::to_string(i);
			params += "' help='Light Position' ";
			TwAddVarCB(bar, name.c_str(), twIEVector3Type,
					   SetLightPos,
					   GetLightPos,
					   &(twCallbackLookup.back()),
					   params.c_str());

			name = "lRadius" + std::to_string(i);
			params = " label='Radius' group='Light#" + std::to_string(i);
			params += "' help='Effecting Radius' ";
			TwAddVarCB(bar, name.c_str(), TW_TYPE_FLOAT,
					   SetLightRadius,
					   GetLightRadius,
					   &(twCallbackLookup.back()),
					   params.c_str());
		}

		params = " Ligths/Light#" + std::to_string(i);
		params+= " group = 'Lights' ";
		TwDefine(params.c_str()); 
		params = " Ligths/Light#" + std::to_string(i);

		if(i == 0)
			params += " opened=true ";
		else
			params += " opened=false ";
		TwDefine(params.c_str());
	}
	TwDefine(" Ligths size='220 250' ");
	TwDefine(" Ligths valueswidth=fit ");
	TwDefine(" Ligths position='5 25' ");
}

void EmptyGISolution::Release()
{
	// Release Tweakbar
	if(bar) TwDeleteBar(bar);
	bar = nullptr;
}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	IEVector3 aColor = ambientLighting ? ambientColor : IEVector3::ZeroVector;
	dRenderer.Render(*currentScene, mainRenderCamera, directLighting, aColor);
}

void EmptyGISolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}