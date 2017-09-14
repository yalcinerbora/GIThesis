#pragma once

#include "IEUtility/IEVector3.h"
#include "SceneLights.h"
#include "AntBar.h"

class LightBar : public AntBar
{
	private:
		struct TwLightCallbackLookup
		{
			uint32_t						lightID;
			SceneLights&					lights;
		};

		static constexpr char*				LightBarName = "Lights";

		static void TW_CALL					GetLightType(void *value, void *clientData);

		static void TW_CALL					GetLightShadow(void *value, void *clientData);
		static void TW_CALL					SetLightShadow(const void *value, void *clientData);

		static void TW_CALL					GetLightColor(void *value, void *clientData);
		static void TW_CALL					SetLightColor(const void *value, void *clientData);

		static void TW_CALL					GetLightIntensity(void *value, void *clientData);
		static void TW_CALL					SetLightIntensity(const void *value, void *clientData);

		static void TW_CALL					GetLightPos(void *value, void *clientData);
		static void TW_CALL					SetLightPos(const void *value, void *clientData);

		static void TW_CALL					GetLightDirection(void *value, void *clientData);
		static void TW_CALL					SetLightDirection(const void *value, void *clientData);

		static void TW_CALL					GetLightRadius(void *value, void *clientData);
		static void TW_CALL					SetLightRadius(const void *value, void *clientData);

		std::vector<TwLightCallbackLookup>	twCallbackLookup;

	public:
		// Constructors & Destructor
											LightBar() = default;
											LightBar(SceneLights& sceneLights,
													 bool& directLightOn,
													 bool& ambientLightOn,
													 IEVector3& ambientColor);
		LightBar&							operator=(LightBar&&) = default;
											~LightBar() = default;

		void								CollapseLights(bool collapsed);

};