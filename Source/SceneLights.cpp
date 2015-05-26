#include "SceneLights.h"
#include "IEUtility/IEVector3.h"

SceneLights::SceneLights(const Array32<Light>& lights)
	: lightsGPU(lights.length)
{
	
}

SceneLights::~SceneLights()
{
	
}

void SceneLights::GenerateShadowMaps(DrawBuffer&, GPUBuffer&)
{
	
}

void SceneLights::ChangeLightPos(uint32_t index, IEVector3 position)
{

}

void SceneLights::ChangeLightType(uint32_t index, LightType)
{

}

void SceneLights::ChangeLightDir(uint32_t index, IEVector3 direction)
{

}

void SceneLights::ChangeLightColor(uint32_t index, IEVector3 color)
{

}

void SceneLights::ChangeLightRadius(uint32_t index, float radius)
{

}
