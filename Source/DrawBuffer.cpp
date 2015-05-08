#include "DrawBuffer.h"
#include "GPUBuffer.h"
#include "IEUtility/IEMatrix4x4.h"
#include "Globals.h"

uint32_t DrawBuffer::initialCapacity = 512;

DrawBuffer::DrawBuffer()
	: drawPoints(initialCapacity)
	, drawTransforms(initialCapacity)
	, drawAABBs(initialCapacity)
{}

void DrawBuffer::AddMaterial(const ColorMaterial& c)
{
	materials.emplace_back(c);
}

void DrawBuffer::AddDrawCall(const DrawPointIndexed& dp,
							 uint32_t mIndex,
							 const ModelTransform& modelTransform,
							 const AABBData& aabb)
{

	drawTransforms.AddData(modelTransform);
	drawPoints.AddData(dp);
	drawAABBs.AddData(aabb);
	materialIndex.push_back(mIndex);
}

StructuredBuffer<ModelTransform>& DrawBuffer::getModelTransformBuffer()
{
	return drawTransforms;
}

StructuredBuffer<AABBData>& DrawBuffer::getAABBBuffer()
{
	return drawAABBs;
}

StructuredBuffer<DrawPointIndexed>& DrawBuffer::getDrawParamBuffer()
{
	return drawPoints;
}

void DrawBuffer::BindMaterialForDraw(uint32_t meshIndex)
{
	materials[materialIndex[meshIndex]].BindMaterial();
}