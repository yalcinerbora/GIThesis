#include "DrawBuffer.h"
#include "GPUBuffer.h"
#include "IEUtility/IEMatrix4x4.h"
#include "Globals.h"

uint32_t DrawBuffer::initialCapacity = 1024;

DrawBuffer::DrawBuffer()
	: drawPoints(initialCapacity)
	, drawTransforms(initialCapacity)
	, drawAABBs(initialCapacity)
	, modelTransformIndices(initialCapacity)
{}

void DrawBuffer::AddMaterial(const ColorMaterial& c)
{
	materials.emplace_back(c);
}

void DrawBuffer::AddTransform(const ModelTransform& mt)
{
	drawTransforms.AddData(mt);
}

void DrawBuffer::AddDrawCall(const DrawPointIndexed& dp,
							 uint32_t mIndex,
							 uint32_t transIndex,
							 const AABBData& aabb)
{
	drawPoints.AddData(dp);
	drawAABBs.AddData(aabb);
	modelTransformIndices.AddData(transIndex);
	materialIndex.push_back(mIndex);
}

void DrawBuffer::SendToGPU()
{
	drawTransforms.SendData();
	drawAABBs.SendData();
	drawPoints.SendData();
	modelTransformIndices.SendData();
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

StructuredBuffer<uint32_t>& DrawBuffer::getModelTransformIndexBuffer()
{
	return modelTransformIndices;
}

void DrawBuffer::BindMaterialForDraw(uint32_t meshIndex)
{
	materials[materialIndex[meshIndex]].BindMaterial();
}