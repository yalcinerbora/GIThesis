#include "IndirectBar.h"
#include "IEUtility/IEMath.h"

void TW_CALL IndirectBar::SetDiffuseTangent(const void *value, void *clientData)
{
	float& diffTan = *static_cast<float*>(clientData);
	diffTan = std::tan(*static_cast<const float*>(value) * static_cast<float>(IEMathConstants::DegToRadCoef) * 0.5f);
}

void TW_CALL IndirectBar::GetDiffuseTangent(void *value, void *clientData)
{
	float& diffTan = *static_cast<float*>(value);
	diffTan = std::atan(*static_cast<const float*>(clientData)) * static_cast<float>(IEMathConstants::RadToDegCoef) * 2.0f;
}

void TW_CALL IndirectBar::SetSpecularTangent(const void *value, void *clientData)
{
	float& spec = *static_cast<float*>(clientData);
	spec = *static_cast<const float*>(value) * static_cast<float>(IEMathConstants::DegToRadCoef);
}

void TW_CALL IndirectBar::GetSpecularTangent(void *value, void *clientData)
{
	float& spec = *static_cast<float*>(value);
	spec = *static_cast<const float*>(clientData) * static_cast<float>(IEMathConstants::RadToDegCoef);
}

IndirectBar::IndirectBar(IndirectUniforms& iUniforms,
						 bool& specularOn,
						 bool& giOn,
						 bool& aoOn)
	: AntBar(IndirectBarName)
{
	std::string paramString;

	// On off
	TwAddVarRW(bar, "giOn", TW_TYPE_BOOLCPP,
			   &giOn,
			   " label='GI On' help='Cone Tracing GI On off' ");
	TwAddVarRW(bar, "aoOn", TW_TYPE_BOOLCPP,
			   &aoOn,
			   " label='AO On' help='Cone Tracing AO On off' ");
	TwAddVarRW(bar, "specularOn", TW_TYPE_BOOLCPP,
			   &specularOn,
			   " label='Specular Cone' help='Specular Cone On Off' ");
	// Cone Angles Section
	TwAddSeparator(bar, "Cone Angles", NULL);
	paramString = " label='Diffuse' help='Diffuse Cone Angle' precision=2 step=0.01";
	paramString += std::string(" min=") + std::to_string(DiffuseLo) + " max=" + std::to_string(DiffuseHi);
	TwAddVarCB(bar, "diffTan", TW_TYPE_FLOAT,
			   SetDiffuseTangent,
			   GetDiffuseTangent,
			   &iUniforms.diffAngleTanHalf,			  
			   paramString.c_str());
	paramString = " label='Specular Min' help='Specular minimum cone angle' precision=2 step=0.01";
	paramString += std::string(" min=") + std::to_string(SpecularMinRangeLo) + " max=" + std::to_string(SpecularMinRangeHi);
	TwAddVarCB(bar, "specMin", TW_TYPE_FLOAT,
			   SetSpecularTangent,
			   GetSpecularTangent,
			   &iUniforms.specularAngleMin,
			   paramString.c_str());
	paramString = " label='Specular Max' help='Specular maximum cone angle' precision=2 step=0.01";
	paramString += std::string(" min=") + std::to_string(SpecularMaxRangeLo) + " max=" + std::to_string(SpecularMaxRangeHi);
	TwAddVarCB(bar, "specMax", TW_TYPE_FLOAT,
			   SetSpecularTangent,
			   GetSpecularTangent,
			   &iUniforms.specularAngleMax,
			   paramString.c_str());
	TwAddSeparator(bar, NULL, NULL);
	paramString = " label='Sample Ratio' help='Trace sampling ratio between samples for each cones' precision=3 step=0.001";
	paramString += std::string(" min=") + std::to_string(SampleRatioLo) + " max=" + std::to_string(SampleRatioHi);
	TwAddVarRW(bar, "sampleRatio", TW_TYPE_FLOAT,
			   &iUniforms.sampleRatio,
			   paramString.c_str());
	paramString = " label='Start Offset Bias' help='Trace sampling start offset in map units' precision=3 step=0.001";
	paramString += std::string(" min=") + std::to_string(OffsetBiasLo) + " max=" + std::to_string(OffsetBiasHi);
	TwAddVarRW(bar, "startOffset", TW_TYPE_FLOAT,
			   &iUniforms.startOffset,
			   paramString.c_str());
	paramString = " label='Total Distance' help='Total tracing distance in map units' precision=2 step=0.1";
	paramString += std::string(" min=") + std::to_string(TotalDistanceLo) + " max=" + std::to_string(TotalDistanceHi);
	TwAddVarRW(bar, "totalDist", TW_TYPE_FLOAT,
			   &iUniforms.totalDistance,
			   paramString.c_str());
	paramString = " label='AO Intensity' help='Ambient Occlusion Intensity' precision=2 step=0.002";
	paramString += std::string(" min=") + std::to_string(AOIntensityLo) + " max=" + std::to_string(AOIntensityHi);
	TwAddVarRW(bar, "aoIntensity", TW_TYPE_FLOAT,
			   &iUniforms.aoIntensity,
			   paramString.c_str());
	paramString = " label='GI Intensity' help='Global Illumination Intensity' precision=2 step=0.002";
	paramString += std::string(" min=") + std::to_string(GIIntensityLo) + " max=" + std::to_string(GIIntensityHi);
	TwAddVarRW(bar, "giIntensity", TW_TYPE_FLOAT,
			   &iUniforms.giIntensity,
			   paramString.c_str());
	paramString = " label='AO Falloff' help='Ambient Occlusion falloff exponent' precision=3 step=0.001";
	paramString += std::string(" min=") + std::to_string(AOFalloffLo) + " max=" + std::to_string(AOFalloffHi);
	TwAddVarRW(bar, "aoFalloff", TW_TYPE_FLOAT,
			   &iUniforms.aoFalloff,
			   paramString.c_str());

	TwDefine((std::string(IndirectBarName) + " size='220 250' ").c_str());
	TwDefine((std::string(IndirectBarName) + " valueswidth=75 ").c_str());
	TwDefine((std::string(IndirectBarName) + " position='5 366' ").c_str());
}


