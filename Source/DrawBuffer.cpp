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
	std::copy(reinterpret_cast<uint8_t*>(cpuDrawPoints.data()), 
			  reinterpret_cast<uint8_t*>(cpuDrawPoints.data()) + cpuDrawPoints.size(),
			  cpuImage.begin() + drawPointOffset);
	std::copy(reinterpret_cast<uint8_t*>(cpuModelTransforms.data()),
			  reinterpret_cast<uint8_t*>(cpuModelTransforms.data()) + cpuModelTransforms.size(),
			  cpuImage.begin() + modelTransformOffset);
	std::copy(reinterpret_cast<uint8_t*>(cpuAABBs.data()),
			  reinterpret_cast<uint8_t*>(cpuAABBs.data()) + cpuAABBs.size(),
			  cpuImage.begin() + aabbOffset);
	std::copy(reinterpret_cast<uint8_t*>(cpuModelTransformIndices.data()),
			  reinterpret_cast<uint8_t*>(cpuModelTransformIndices.data()) + cpuAABBs.size(),
			  cpuImage.begin() + modelTransformIndexOffset);

	// Finalize and Send
	gpuData.SendData();
}

void DrawBuffer::SendModelTransformToGPU(uint32_t offset, uint32_t size)
{
	assert((offset + size) <= (cpuModelTransforms.size() * sizeof(ModelTransform)));
	if(locked)
	{
		uint32_t subSize = (size == std::numeric_limits<uint32_t>::max()) ?
						   static_cast<uint32_t>(cpuModelTransforms.size()) : size;
		gpuData.SendSubData(static_cast<uint32_t>(modelTransformOffset) +
							offset * sizeof(ModelTransform),
							subSize * sizeof(ModelTransform));
	}
}

ModelTransform& DrawBuffer::ModelTransformBuffer(uint32_t transformId)
{
	return cpuModelTransforms[transformId];
}

GLuint DrawBuffer::getGLBuffer()
{
	return gpuData.getGLBuffer();
}

size_t DrawBuffer::getModelTransformOffset() const
{
	return modelTransformOffset;
}

size_t DrawBuffer::getAABBOffset() const
{
	return aabbOffset;
}

size_t DrawBuffer::getDrawParamOffset() const
{
	return drawPointOffset;
}

size_t DrawBuffer::getModelTransformIndexOffset() const
{
	return modelTransformIndexOffset;
}

const std::vector<DrawPointIndexed>& DrawBuffer::getCPUDrawPoints() const
{
	return cpuDrawPoints;
}

const std::vector<ModelTransform>& DrawBuffer::getCPUModelTransforms() const
{
	return cpuModelTransforms;
}

const std::vector<AABBData>& DrawBuffer::getCPUAABBs() const
{
	return cpuAABBs;
}

const std::vector<uint32_t>& DrawBuffer::getCPUModelTransformIndices() const
{
	return cpuModelTransformIndices;
}

void DrawBuffer::BindAsDrawIndirectBuffer()
{
	gpuData.BindAsDrawIndirectBuffer();
}

void DrawBuffer::BindAABB(GLuint bindPoint)
{
	gpuData.BindAsShaderStorageBuffer(bindPoint, 
									  static_cast<GLuint>(aabbOffset),
									  static_cast<GLuint>(cpuAABBs.size() * sizeof(AABBData)));
}

void DrawBuffer::BindModelTransform(GLuint bindPoint)
{
	gpuData.BindAsShaderStorageBuffer(bindPoint,
									  static_cast<GLuint>(modelTransformOffset),
									  static_cast<GLuint>(cpuModelTransforms.size() * sizeof(ModelTransform)));
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
								static_cast<GLsizei>(cpuDrawPoints.size()),
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