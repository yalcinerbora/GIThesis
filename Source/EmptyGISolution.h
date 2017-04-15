/**

Empty Solution
Just Renders the scene

*/

#ifndef __EMPTYGISOLUTION_H__
#define __EMPTYGISOLUTION_H__

#include "SolutionI.h"
#include <AntTweakBar.h>
#include <cstdint>
#include <vector>
#include "IEUtility/IEVector3.h"

class DeferredRenderer;
class EmptyGISolution;

struct TwLightCallbackLookup
{
	uint32_t			lightID;
	EmptyGISolution*	solution;
};

class EmptyGISolution : public SolutionI
{
	private:
		const std::string			name;
		TwBar*						bar;
		double						frameTime;

		// For Callback Handling
		static std::vector<TwLightCallbackLookup> twCallbackLookup;

		static void TW_CALL			GetLightType(void *value, void *clientData);

		static void TW_CALL			GetLightShadow(void *value, void *clientData);
		static void TW_CALL			SetLightShadow(const void *value, void *clientData);

		static void TW_CALL			GetLightColor(void *value, void *clientData);
		static void TW_CALL			SetLightColor(const void *value, void *clientData);

		static void TW_CALL			GetLightIntensity(void *value, void *clientData);
		static void TW_CALL			SetLightIntensity(const void *value, void *clientData);

		static void TW_CALL			GetLightPos(void *value, void *clientData);
		static void TW_CALL			SetLightPos(const void *value, void *clientData);

		static void TW_CALL			GetLightDirection(void *value, void *clientData);
		static void TW_CALL			SetLightDirection(const void *value, void *clientData);

		static void TW_CALL			GetLightRadius(void *value, void *clientData);
		static void TW_CALL			SetLightRadius(const void *value, void *clientData);

	protected:
		DeferredRenderer&			dRenderer;
		SceneI*						currentScene;

		bool						directLighting;
		bool						ambientLighting;
		IEVector3					ambientColor;
	
	public:
									EmptyGISolution(const std::string& name, 
													DeferredRenderer&);
									~EmptyGISolution() = default;
	
		bool						IsCurrentScene(SceneI&) override;
		void						Load(SceneI&) override;
		void						Release() override;
		void						Frame(const Camera&) override;
		void						SetFPS(double fpsMS) override;

		const std::string&			Name() const override;
};
#endif //__EMPTYGISOLUTION_H__