#include "DrawBuffer.h"
#include "VertexBuffer.h"
#include "IEUtility/IEMatrix4x4.h"

DrawBuffer::DrawBuffer()
	: drawPointOffset(0)
	, modelTransformOffset(0)
	, aabbOffset(0)
	, modelTransformIndexOffset(0)
	, locked(false)
{}

DrawBuffer::DrawBuffer(DrawBuffer&& other)
	: drawPointOffset(other.drawPointOffset)
	, modelTransformOffset(other.modelTransformOffset)
	, aabbOffset(other.aabbOffset)
	, modelTransformIndexOffset(other.modelTransformIndexOffset)
	, locked(other.locked)
	, cpuDrawPoints(std::move(other.cpuDrawPoints))
	, cpuModelTransforms(std::move(other.cpuModelTransforms))
	, cpuAABBs(std::move(other.cpuAABBs))
	, cpuModelTransformIndices(std::move(other.cpuModelTransformIndices))
	, gpuData(std::move(other.gpuData))
	, drawMaterialIndex(std::move(other.drawMaterialIndex))
	, materials(std::move(other.materials))
{}

DrawBuffer& DrawBuffer::operator=(DrawBuffer&& other)
{
	drawPointOffset = other.drawPointOffset;
	modelTransformOffset = other.modelTransformOffset;
	aabbOffset = other.aabbOffset;
	modelTransformIndexOffset = other.modelTransformIndexOffset;
	locked = other.locked;
	cpuDrawPoints = std::move(other.cpuDrawPoints);
	cpuModelTransforms = std::move(other.cpuModelTransforms);
	cpuAABBs = std::move(other.cpuAABBs);
	cpuModelTransformIndices = std::move(other.cpuModelTransformIndices);
	gpuData = std::move(other.gpuData);
	drawMaterialIndex = std::move(other.drawMaterialIndex);
	materials = std::move(other.materials);
	return *this;
}

uint32_t DrawBuffer::AddMaterial(const ColorMaterial& c)
{
	assert(!locked);
	materials.emplace_back(c);
	return static_cast<uint32_t>(materials.size() - 1);
}

uint32_t DrawBuffer::AddTransform(const ModelTransform& mt)
{
	assert(!locked);
	cpuModelTransforms.push_back(mt);
	return static_cast<uint32_t>(cpuModelTransforms.size() - 1);
}

uint32_t DrawBuffer::AddDrawCall(const DrawPointIndexed& dp,
								 uint32_t mIndex,
								 uint32_t transIndex,
								 const AABBData& aabb)
{
	assert(!locked);
	cpuDrawPoints.push_back(dp);
	cpuAABBs.push_back(aabb);
	cpuModelTransformIndices.push_back(transIndex);
	drawMaterialIndex.push_back(mIndex);
	return static_cast<uint32_t>(cpuDrawPoints.size() - 1);
}

void DrawBuffer::LockAndLoad()
{
	locked = true;
	size_t totalSize = 0;

	// DP
	totalSize = DeviceOGLParameters::AlignOffset(totalSize, 4);
	drawPointOffset = totalSize;
	totalSize += cpuDrawPoints.size() * sizeof(DrawPointIndexed);
	// Model Trans
	totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	modelTransformOffset = totalSize;
	totalSize += cpuModelTransforms.size() * sizeof(ModelTransform);
	// AABB
	totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	aabbOffset = totalSize;
	totalSize += cpuAABBs.size() * sizeof(AABBData);
	// Transform Indices
	//totalSize = DeviceOGLParameters::SSBOAlignOffset(totalSize);
	modelTransformIndexOffset = totalSize;
	totalSize += cpuModelTransformIndices.size() * sizeof(uint32_t);

	// Copy Data
	gpuData.Resize(totalSize);
	auto& cpuImage = gpuData.CPUData();
	std::copy(reinterpret_cast<uint8_t*>(cpuDrawPoints.data()), 
			  reinterpret_cast<uint8_t*>(cpuDrawPoints.data() + cpuDrawPoints.size()),
			  cpuImage.data() + drawPointOffset);
	std::copy(reinterpret_cast<uint8_t*>(cpuModelTransforms.data()),
			  reinterpret_cast<uint8_t*>(cpuModelTransforms.data() + cpuModelTransforms.size()),
			  cpuImage.data() + modelTransformOffset);
	std::copy(reinterpret_cast<uint8_t*>(cpuAABBs.data()),
			  reinterpret_cast<uint8_t*>(cpuAABBs.data() + cpuAABBs.size()),
			  cpuImage.data() + aabbOffset);
	std::copy(reinterpret_cast<uint8_t*>(cpuModelTransformIndices.data()),
			  reinterpret_cast<uint8_t*>(cpuModelTransformIndices.data() + cpuModelTransformIndices.size()),
			  cpuImage.data() + modelTransformIndexOffset);

	// Finalize and Send
	gpuData.SendData();
}

void DrawBuffer::SendModelTransformToGPU(uint32_t offset, uint32_t size)
{
	assert(locked);
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
	assert(locked);
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

size_t DrawBuffer::getDrawPointCount() const
{
	return cpuDrawPoints.size();
}

size_t DrawBuffer::getModelTransformCount() const
{
	return cpuModelTransforms.size();
}

size_t DrawBuffer::getAABBCount() const
{
	return cpuAABBs.size();
}

size_t DrawBuffer::getModelTransformIndexCount() const
{
	return cpuModelTransformIndices.size();
}

size_t DrawBuffer::getMaterialCount() const
{
	return materials.size();
}

const AABBData& DrawBuffer::getAABB(uint32_t drawId) const
{
	return cpuAABBs[drawId];
}

//const std::vector<DrawPointIndexed>& DrawBuffer::getCPUDrawPoints() const
//{
//	return cpuDrawPoints;
//}
//
//const std::vector<ModelTransform>& DrawBuffer::getCPUModelTransforms() const
//{
//	return cpuModelTransforms;
//}
//
//const std::vector<AABBData>& DrawBuffer::getCPUAABBs() const
//{
//	return cpuAABBs;
//}
//
//const std::vector<uint32_t>& DrawBuffer::getCPUModelTransformIndices() const
//{
//	return cpuModelTransformIndices;
//}
//
//const std::vector<Material>& DrawBuffer::getMaterials() const
//{
//	return materials;
//}

void DrawBuffer::BindAsDrawIndirectBuffer()
{
	assert(locked);
	gpuData.BindAsDrawIndirectBuffer();
}

void DrawBuffer::BindAABB(GLuint bindPoint)
{
	assert(locked);
	gpuData.BindAsShaderStorageBuffer(bindPoint, 
									  static_cast<GLuint>(aabbOffset),
									  static_cast<GLuint>(cpuAABBs.size() * sizeof(AABBData)));
}

void DrawBuffer::BindModelTransform(GLuint bindPoint)
{
	assert(locked);
	gpuData.BindAsShaderStorageBuffer(bindPoint,
									  static_cast<GLuint>(modelTransformOffset),
									  static_cast<GLuint>(cpuModelTransforms.size() * sizeof(ModelTransform)));
}

void DrawBuffer::BindMaterialForDraw(uint32_t meshIndex)
{
	assert(locked);
	materials[drawMaterialIndex[meshIndex]].BindMaterial();
}

void DrawBuffer::DrawCallSingle(GLuint drawId)
{
	assert(locked);
	static_assert(sizeof(GLintptr) == sizeof(void*), "Unappropirate GL Offset Parameter");
	GLintptr offset = static_cast<GLintptr>(drawId * sizeof(DrawPointIndexed));
	glDrawElementsIndirect(GL_TRIANGLES,
						   GL_UNSIGNED_INT,
						   (void *)(offset));
}

void DrawBuffer::DrawCallMulti()
{
	assert(locked);
	glMultiDrawElementsIndirect(GL_TRIANGLES,
								GL_UNSIGNED_INT,
								nullptr,
								static_cast<GLsizei>(cpuDrawPoints.size()),
								sizeof(DrawPointIndexed));
}

void DrawBuffer::DrawCallMultiState()
{
	assert(locked);
	for(int i = 0; i < cpuDrawPoints.size(); i++)
	{
		static_assert(sizeof(GLintptr) == sizeof(void*), "Unappropirate GL Offset Parameter");
		GLintptr offset = static_cast<GLintptr>(i * sizeof(DrawPointIndexed));		
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (offset));
	}
}