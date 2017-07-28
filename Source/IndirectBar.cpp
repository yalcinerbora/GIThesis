#include "IndirectBar.h"

IndirectBar::IndirectBar(SceneLights& sceneLights,
						 IndirectUniforms& iUniforms,
						 bool& specularOn)
	: AntBar("ConeParams")
{

}

//// On off
//TwAddSeparator(bar, NULL, NULL);	
//TwAddVarRW(bar, "giOn", TW_TYPE_BOOLCPP,
//		   &giOn,
//		   " label='GI On' help='Cone Tracing GI On off' ");
//TwAddVarRW(bar, "aoOn", TW_TYPE_BOOLCPP,
//		   &aoOn,
//		   " label='AO On' help='Cone Tracing AO On off' ");
//TwAddVarRW(bar, "inject", TW_TYPE_BOOLCPP,
//		   &injectOn,
//		   " label='Inject' help='Light Inject On Off' ");
