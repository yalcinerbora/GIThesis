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
	, light(0)
	, lightLevel(0)
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
		light = std::min(totalLight - 1, light + 1);
		lightLevel = std::min(lightMaxLevel[light], lightLevel);
	}
}

void RenderSelect::Previous()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		light = std::max(0, light - 1);
		lightLevel = std::min(lightMaxLevel[light], lightLevel);
	}
}

void RenderSelect::Up()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
		lightLevel = std::min(lightMaxLevel[light] - 1, lightLevel + 1);
}

void RenderSelect::Down()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
		lightLevel = std::max(0, lightLevel - 1);
}

int RenderSelect::Light() const
{
	return light;
}

int RenderSelect::LightLevel() const
{
	return lightLevel;
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
	, svoLevel(maxSVOLevel)
	, cacheCascade(0)
	, pageCascade(0)
	, svoRenderType(OctreeRenderType::IRRADIANCE)
	, cacheRenderType(VoxelRenderType::DIFFUSE_ALBEDO)
	, pageRenderType(VoxelRenderType::DIFFUSE_ALBEDO)
{}

void VoxelRenderSelect::Next()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		RenderSelect::Next();
	}
	else if(*scheme == RenderScheme::VOXEL_CACHE)
	{
		int renderType = static_cast<int>(cacheRenderType);
		int vTypeMax = static_cast<int>(VoxelRenderType::END) - 1;
		cacheRenderType = static_cast<VoxelRenderType>(std::min(vTypeMax, renderType + 1));
	}
	else if(*scheme == RenderScheme::VOXEL_PAGE)
	{
		int renderType = static_cast<int>(pageRenderType);
		int vTypeMax = static_cast<int>(VoxelRenderType::END) - 1;
		pageRenderType = static_cast<VoxelRenderType>(std::min(vTypeMax, renderType + 1));
	}
	else if(*scheme == RenderScheme::SVO_VOXELS ||
			*scheme == RenderScheme::SVO_SAMPLE)
	{
		int renderType = static_cast<int>(svoRenderType);
		int sTypeMax = static_cast<int>(VoxelRenderType::END) - 1;
		svoRenderType = static_cast<OctreeRenderType>(std::min(sTypeMax, renderType + 1));
	}
}

void VoxelRenderSelect::Previous()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		RenderSelect::Previous();
	}
	else if(*scheme == RenderScheme::VOXEL_CACHE)
	{
		int renderType = static_cast<int>(cacheRenderType);
		cacheRenderType = static_cast<VoxelRenderType>(std::max(0, renderType - 1));
	}
	else if(*scheme == RenderScheme::VOXEL_PAGE)
	{
		int renderType = static_cast<int>(pageRenderType);
		pageRenderType = static_cast<VoxelRenderType>(std::max(0, renderType - 1));
	}
	else if(*scheme == RenderScheme::SVO_VOXELS ||
			*scheme == RenderScheme::SVO_SAMPLE)
	{
		int renderType = static_cast<int>(svoRenderType);
		svoRenderType = static_cast<OctreeRenderType>(std::max(0, renderType - 1));
	}
}

void VoxelRenderSelect::Up()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		RenderSelect::Up();
	}
	else if(*scheme == RenderScheme::VOXEL_CACHE)
	{
		cacheCascade = std::min(totalCascades - 1, cacheCascade + 1);
	}
	else if(*scheme == RenderScheme::VOXEL_PAGE)
	{
		pageCascade = std::min(totalCascades - 1, pageCascade + 1);
	}
	else if(*scheme == RenderScheme::SVO_VOXELS ||
			*scheme == RenderScheme::SVO_SAMPLE)
	{
		svoLevel = std::min(maxSVOLevel, svoLevel + 1);
	}
}

void VoxelRenderSelect::Down()
{
	if(*scheme == RenderScheme::SHADOW_MAP)
	{
		RenderSelect::Down();
	}
	else if(*scheme == RenderScheme::VOXEL_CACHE)
	{
		cacheCascade = std::max(0, cacheCascade - 1);
	}
	else if(*scheme == RenderScheme::VOXEL_PAGE)
	{
		pageCascade = std::max(0, pageCascade - 1);
	}
	else if(*scheme == RenderScheme::SVO_VOXELS ||
			*scheme == RenderScheme::SVO_SAMPLE)
	{
		svoLevel = std::max(minSVOLevel, svoLevel - 1);
	}
}

int VoxelRenderSelect::SVOLevel() const
{
	return svoLevel;
}

OctreeRenderType VoxelRenderSelect::SVORenderType() const
{
	return svoRenderType;
}

int VoxelRenderSelect::CacheCascade() const
{
	return cacheCascade;
}

VoxelRenderType VoxelRenderSelect::CacheRenderType() const
{
	return cacheRenderType;
}

int VoxelRenderSelect::PageCascade() const
{
	return pageCascade;
}

VoxelRenderType VoxelRenderSelect::PageRenderType() const
{
	return pageRenderType;
}