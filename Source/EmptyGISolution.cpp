#include "EmptyGISolution.h"
#include "Camera.h"
#include "SceneI.h"
#include "DeferredRenderer.h"
#include "Macros.h"
#include "SceneLights.h"

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
	*static_cast<IEVector3*>(value) = color.NormalizeSelf();
}

void TW_CALL EmptyGISolution::SetLightColor(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 color = lookup->solution->currentScene->getSceneLights().GetLightColor(lookup->lightID);
	float intensity = color.Length();
	lookup->solution->currentScene->getSceneLights().ChangeLightColor(lookup->lightID,
																	  intensity * (*static_cast<const IEVector3*>(value)));
}
	    
void TW_CALL EmptyGISolution::GetLightIntensity(void *value, void *clientData)
{

}

void TW_CALL EmptyGISolution::SetLightIntensity(const void *value, void *clientData)
{

}
	    
void TW_CALL EmptyGISolution::GetLightPos(void *value, void *clientData)
{

}

void TW_CALL EmptyGISolution::SetLightPos(const void *value, void *clientData)
{

}

void TW_CALL EmptyGISolution::GetLightDirection(void *value, void *clientData)
{

}

void TW_CALL EmptyGISolution::SetLightDirection(const void *value, void *clientData)
{

}

EmptyGISolution::EmptyGISolution(DeferredRenderer& defferedRenderer)
	: currentScene(nullptr)
	, dRenderer(defferedRenderer)
	, bar(nullptr)
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
	bar = TwNewBar("EmptyGI");
	TwDefine(" EmptyGI refresh=0.01 ");

	// FPS Show
	TwAddVarRO(bar, "fTime", TW_TYPE_DOUBLE, &frameTime,
			   " label='Frame(ms)' help='Frame Time in milliseconds..' ");
	TwAddSeparator(bar, NULL, NULL);
		
	std::string name;
	std::string params;
	twCallbackLookup.reserve(s.getSceneLights().Count());
	twCallbackLookup.clear();
	for(unsigned int i = 0; i < s.getSceneLights().Count(); i++)
	{
		twCallbackLookup.push_back({ i, this });

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

		//name = "lIntensity" + std::to_string(i);
		//params = " label='Color' group='Light#" + std::to_string(i);
		//params += "' help='Light Color' ";
		//TwAddVarCB(bar, name.c_str(), TW_TYPE_COLOR3F,
		//		   NULL,//SetLightColor,
		//		   GetLightColor,
		//		   &(twCallbackLookup.back()),
		//		   params.c_str());


		params = " EmptyGI/Light#" + std::to_string(i);
		params+= " group = 'Lights' ";
		TwDefine(params.c_str()); 
		// The group 'Background' of bar 'Main' is put in the group 'Display'
	}

}

void EmptyGISolution::Release()
{
	// Release Tweakbar
	if(bar) TwDeleteBar(bar);
}

void EmptyGISolution::Frame(const Camera& mainRenderCamera)
{
	dRenderer.Render(*currentScene, mainRenderCamera);
}

void EmptyGISolution::SetFPS(double fpsMS)
{
	frameTime = fpsMS;
}