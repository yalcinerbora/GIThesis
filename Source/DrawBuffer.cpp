#include "DrawBuffer.h"
#include "VertexBuffer.h"
#include "IEUtility/IEMatrix4x4.h"
#include "Globals.h"

DrawBuffer::DrawBuffer()
	: drawPointOffset(0)
	, modelTransformOffset(0)
	, aabbOffset(0)
	, modelTransformIndexOffset(0)
	, locked(false)
{}

uint32_t DrawBuffer::AddMaterial(const ColorMaterial& c)
{
	materials.emplace_back(c);
	return static_cast<uint32_t>(materials.size() - 1);
}

uint32_t DrawBuffer::AddTransform(const ModelTransform& mt)
{
	cpuModelTransforms.push_back(mt);
	return static_cast<uint32_t>(cpuModelTransforms.size() - 1);
}

uint32_t DrawBuffer::AddDrawCall(const DrawPointIndexed& dp,
								 uint32_t mIndex,
								 uint32_t transIndex,
								 const AABBData& aabb)
{
	cpuDrawPoints.push_back(dp);
	cpuAABBs.push_back(aabb);
	cpuModelTransformIndices.push_back(transIndex);
	drawMaterialIndex.push_back(mIndex);
}

void DrawBuffer::LockAndLoad()
{
	locked = true;
	size_t totalSize = 0;

	// DP
	drawPointOffset = totalSize;
	totalSize += cpuDrawPoints.size() * sizeof(DrawPointIndexed);
	// Model Trans
	totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	modelTransformOffset = totalSize;
	totalSize += cpuModelTransforms.size() * sizeof(DrawPointIndexed);
	// AABB
	totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	aabbOffset = totalSize;
	totalSize += cpuAABBs.size() * sizeof(AABBData);
	// Transform Indices
	totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	modelTransformIndexOffset = totalSize;
	totalSize += cpuModelTransformIndices.size() * sizeof(uint32_t);

	auto& cpuImage = gpuData.CPUData();
	cpuImage.resize(totalSize);

	// Copy Data
	std::copy(cpuDrawPoints.begin(), cpuDrawPoints.end(),
			  cpuImage.begin() + drawPointOffset);
	std::copy(cpuModelTransforms.begin(), cpuModelTransforms.end(),
			  cpuImage.begin() + modelTransformOffset);
	std::copy(cpuAABBs.begin(), cpuAABBs.end(),
			  cpuImage.begin() + aabbOffset);
	std::copy(cpuModelTransformIndices.begin(), cpuModelTransformIndices.end(),
			  cpuImage.begin() + modelTransformIndexOffset);

	// Finalize and Send
	gpuData.SendData();
}

void DrawBuffer::SendModelTransformToGPU(uint32_t offset = 0, uint32_t size = std::numeric_limits<uint32_t>::max())
{
	assert(offset + size <= cpuModelTransforms.size * sizeof(ModelTransform));
	if(locked)
	{
		uint32_t subSize = (size == std::numeric_limits<uint32_t>::max()) ?
						   cpuModelTransforms.size() : size;
		gpuData.SendSubData(modelTransformOffset +
							offset * sizeof(ModelTransform),
							subSize * sizeof(ModelTransform));
	}
}

ModelTransform& DrawBuffer::ModelTransformBuffer(uint32_t transformId)
{
	return cpuModelTransforms[transformId];
}

void DrawBuffer::BindAsDrawIndirectBuffer()
{
	gpuData.BindAsDrawIndirectBuffer();
}

void DrawBuffer::BindAABB(GLuint bindPoint)
{
	gpuData.BindAsShaderStorageBuffer(bindPoint, aabbOffset,
									  cpuAABBs.size() * sizeof(AABBData));
}

void DrawBuffer::BindModelTransform(GLuint bindPoint)
{
	gpuData.BindAsShaderStorageBuffer(bindPoint,
									  modelTransformOffset,
									  cpuModelTransforms.size() * sizeof(ModelTransform));
}

void DrawBuffer::BindMaterialForDraw(uint32_t meshIndex)
{
	materials[drawMaterialIndex[meshIndex]].BindMaterial();
}

void DrawBuffer::DrawCallSingle(GLuint drawId)
{
	glDrawElementsIndirect(GL_TRIANGLES,
						   GL_UNSIGNED_INT,
						   (void *)(drawId * sizeof(DrawPointIndexed)));
}

void DrawBuffer::DrawCallMulti()
{
	glMultiDrawElementsIndirect(GL_TRIANGLES,
								GL_UNSIGNED_INT,
								nullptr,
								cpuDrawPoints.size(),
								sizeof(DrawPointIndexed));
}

void DrawBuffer::DrawCallMultiState()
{
	for(int i = 0; i < cpuDrawPoints.size(); i++)
	{
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));
	}
}