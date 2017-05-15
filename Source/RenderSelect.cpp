#include "RenderSelect.h"
#include "SceneLights.h"
#include "DeferredRenderer.h"

const TwEnumVal RenderSelect::renderSchemeVals[] =
{ 
	{static_cast<int>(RenderScheme::FINAL), "Final"},
	{static_cast<int>(RenderScheme::LIGHT_INTENSITY), "Light Intesity"},
	{static_cast<int>(RenderScheme::G_DIFF_ALBEDO), "Diffuse Albedo"},
	{static_cast<int>(RenderScheme::G_SPEC_ALBEDO), "Specular Albedo"},
	{static_cast<int>(RenderScheme::G_NORMAL), "Normals"},
	{static_cast<int>(RenderScheme::G_DEPTH), "Depth"},
	{static_cast<int>(RenderScheme::SHADOW_MAP), "Shadow Maps"},
	{static_cast<int>(RenderScheme::SVO_SAMPLE), "SVO-Scene Aligment"},
	{static_cast<int>(RenderScheme::SVO_VOXELS), "SVO Voxels"},
	{static_cast<int>(RenderScheme::VOXEL_PAGE), "Page Voxels"},
	{static_cast<int>(RenderScheme::VOXEL_CACHE), "Cache Voxels"}
};

TwType RenderSelect::twDeferredRenderType = TwType::TW_TYPE_UNDEF;

void RenderSelect::GenRenderTypeEnum()
{
	twDeferredRenderType = TwDefineEnum("Render Type", renderSchemeVals, 
										static_cast<unsigned int>(RenderScheme::SVO_SAMPLE));
}

RenderSelect::RenderSelect(TwBar* bar, const SceneLights& lights,
						   RenderScheme& scheme)
	: RenderSelect(bar, lights, scheme, twDeferredRenderType)
{}

RenderSelect::RenderSelect(TwBar* bar,
						   const SceneLights& lights,
						   RenderScheme& scheme,
						   TwType enumType)
	: totalLight(static_cast<int>(lights.getLightCount()))
	, currentLight(0)
	, currentLightLevel(0)
	, scheme(&scheme)
{
	TwAddVarRW(bar, "renderType", enumType,
			   &scheme,
			   " label='Render' help='Change what to show on screen' ");

	lightMaxLevel.resize(totalLight);
	for(int i = 0; i < totalLight; i++)
	{
		if(lights.getLightType(i) == LightType::POINT)
		{
			lightMaxLevel[i] = SceneLights::CubeSide;
		}
		else if(lights.getLightType(i) == LightType::DIRECTIONAL)
		{
			lightMaxLevel[i] = LightDrawBuffer::DirectionalCascadesCount;
		}
	}
}

void RenderSelect::Next()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		currentLight = std::min(totalLight - 1, currentLight + 1);
		currentLightLevel = std::min(lightMaxLevel[currentLight], currentLightLevel);
	}
}

void RenderSelect::Previous()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		currentLight = std::max(0, currentLight - 1);
		currentLightLevel = std::min(lightMaxLevel[currentLight], currentLightLevel);
	}
}

void RenderSelect::Up()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
		currentLightLevel = std::min(lightMaxLevel[currentLight] - 1, currentLightLevel + 1);
}

void RenderSelect::Down()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
		currentLightLevel = std::max(0, currentLightLevel - 1);
}

int RenderSelect::CurrentLight() const
{
	return currentLight;
}

int RenderSelect::CurrentLightLevel() const
{
	return currentLightLevel;
}

TwType VoxelRenderSelect::twThesisRenderType = TwType::TW_TYPE_UNDEF;

void VoxelRenderSelect::GenRenderTypeEnum()
{
	twThesisRenderType = TwDefineEnum("Voxel Render Type", renderSchemeVals, 
									  static_cast<unsigned int>(RenderScheme::END));
}

VoxelRenderSelect::VoxelRenderSelect(TwBar* bar, const SceneLights& lights, 
									 RenderScheme& scheme,
									 int totalCascades,
									 int minSVOLevel, int maxSVOLevel)
	: RenderSelect(bar, lights, scheme, twThesisRenderType)
	, totalCascades(totalCascades)
	, minSVOLevel(minSVOLevel)
	, maxSVOLevel(maxSVOLevel)
	, currentSVOLevel(0)
	, currentCacheCascade(0)
	, currentPageLevel(0)
{}

void VoxelRenderSelect::Next()
{

}

void VoxelRenderSelect::Previous()
{

}

void VoxelRenderSelect::Up()
{

}

void VoxelRenderSelect::Down()
{

}
