#include "LightBar.h"

void TW_CALL LightBar::GetLightType(void *value, void *clientData)
{
	static const char* names[] =
	{
		"POINT",
		"DIRECTIONAL",
		"AREA",
	};
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	LightType t = lookup->lights.getLightType(lookup->lightID);
	*static_cast<const char**>(value) = names[static_cast<int>(t)];
}

void TW_CALL LightBar::GetLightShadow(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	*static_cast<bool*>(value) = lookup->lights.getLightCastShadow(lookup->lightID);
}

void TW_CALL LightBar::SetLightShadow(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->lights.ChangeLightShadow(lookup->lightID,
									  *(static_cast<const bool*>(value)));
}

void TW_CALL LightBar::GetLightColor(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 color = lookup->lights.getLightColor(lookup->lightID);
	*static_cast<IEVector3*>(value) = color;
}

void TW_CALL LightBar::SetLightColor(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->lights.ChangeLightColor(lookup->lightID,
		(*static_cast<const IEVector3*>(value)));
}

void TW_CALL LightBar::GetLightIntensity(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	float intensity = lookup->lights.getLightIntensity(lookup->lightID);
	*static_cast<float*>(value) = intensity;
}

void TW_CALL LightBar::SetLightIntensity(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->lights.ChangeLightIntensity(lookup->lightID,
		(*static_cast<const float*>(value)));
}

void TW_CALL LightBar::GetLightPos(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 pos = lookup->lights.getLightPos(lookup->lightID);
	*static_cast<IEVector3*>(value) = pos;
}

void TW_CALL LightBar::SetLightPos(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->lights.ChangeLightPos(lookup->lightID,
		(*static_cast<const IEVector3*>(value)));
}

void TW_CALL LightBar::GetLightDirection(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	IEVector3 intensity = lookup->lights.getLightDir(lookup->lightID);
	*static_cast<IEVector3*>(value) = intensity.Normalize();
}

void TW_CALL LightBar::SetLightDirection(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->lights.ChangeLightDir(lookup->lightID,
		(*static_cast<const IEVector3*>(value)));
}

void TW_CALL LightBar::GetLightRadius(void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	float radius = lookup->lights.getLightRadius(lookup->lightID);
	*static_cast<float*>(value) = radius;
}

void TW_CALL LightBar::SetLightRadius(const void *value, void *clientData)
{
	TwLightCallbackLookup* lookup = static_cast<TwLightCallbackLookup*>(clientData);
	lookup->lights.ChangeLightRadius(lookup->lightID,
		(*static_cast<const float*>(value)));
}

LightBar::LightBar(SceneLights& sceneLights,
				   bool& directLightOn,
				   bool& ambientLightOn,
				   IEVector3& ambientColor)
	: AntBar(LightBarName)
{
	// Generic
	TwAddVarRW(bar, "directLightOn", TW_TYPE_BOOLCPP,
			   &directLightOn,
			   " label='Direct Light' help='Direct Ligting On Off' ");
	TwAddVarRW(bar, "ambientOn", TW_TYPE_BOOLCPP,
			   &ambientLightOn,
			   " label='Ambient On' help='Ambient Ligting On Off' ");
	TwAddVarRW(bar, "aColor", TW_TYPE_COLOR3F,
			   &ambientColor,
			   " label='Ambient Color' help='Ambient Color'");

	TwAddSeparator(bar, NULL, NULL);

	std::string name;
	std::string params;
	twCallbackLookup.reserve(sceneLights.getLightCount());
	twCallbackLookup.clear();
	for(unsigned int i = 0; i < sceneLights.getLightCount(); i++)
	{
		twCallbackLookup.push_back({i, sceneLights});
		LightType lightType = sceneLights.getLightType(i);

		name = "lType" + std::to_string(i);
		params = " label='Type' group='Light#" + std::to_string(i);
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

		if(lightType == LightType::DIRECTIONAL)
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

		if(lightType == LightType::POINT)
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

		params = std::string(LightBarName) + "/Light#" + std::to_string(i);
		params += " group = 'Lights' ";
		TwDefine(params.c_str());
		params = std::string(LightBarName) + "/Light#" + std::to_string(i);

		if(i == 0)
			params += " opened=true ";
		else
			params += " opened=false ";
		TwDefine(params.c_str());
	}
	TwDefine((std::string(LightBarName) + " size='220 250' ").c_str());
	TwDefine((std::string(LightBarName) + " valueswidth=fit ").c_str());
	TwDefine((std::string(LightBarName) + " position='5 25' ").c_str());
}