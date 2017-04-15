#include "VertexBuffer.h"
#include "Macros.h"
#include "Globals.h"
#include "GFG/GFGMeshHeader.h"
#include <algorithm>

VertexBuffer::VertexBuffer(const std::vector<VertexElement>& elements,
						   uint32_t byteStride)
	: vao(0)
	, byteStride(byteStride)
	, vElements(elements.begin(), elements.end())
	, addLocked(false)
{}

VertexBuffer::~VertexBuffer()
{
	glDeleteVertexArrays(1, &vao);
}

bool VertexBuffer::GFGLogicHit(GFGVertexComponentLogic gfg, VertexLogic l)
{
	bool sameLogic = false; 
	sameLogic |= (gfg == GFGVertexComponentLogic::POSITION) &&
				 (l == VertexLogic::POSITION);
	sameLogic |= (gfg == GFGVertexComponentLogic::NORMAL) &&
				 (l == VertexLogic::NORMAL);
	sameLogic |= (gfg == GFGVertexComponentLogic::UV) &&
				 (l == VertexLogic::UV);
	sameLogic |= (gfg == GFGVertexComponentLogic::WEIGHT) &&
				 (l == VertexLogic::WEIGHT);
	sameLogic |= (gfg == GFGVertexComponentLogic::WEIGHT_INDEX) &&
				 (l == VertexLogic::WEIGHT_INDEX);
	return sameLogic;
}

bool VertexBuffer::GFGSupportedLogic(GFGVertexComponentLogic gfg)
{
	bool supported = true;
	supported &= (gfg != GFGVertexComponentLogic::TANGENT);
	supported &= (gfg != GFGVertexComponentLogic::BINORMAL);
	supported &= (gfg != GFGVertexComponentLogic::COLOR);
	return supported;
}

bool VertexBuffer::GFGSameDataType(GFGDataType gfg, GPUDataType type, uint32_t typeCount)
{
	if(typeCount > 4) return false;

	bool sameType = true;
	uint32_t dataTypeStart = 0;
	switch(type)
	{
		case GPUDataType::INT8:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::INT8_1);
			break;
		case GPUDataType::INT16:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::INT16_1);
			break;
		case GPUDataType::INT32:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::INT32_1);
			break;
		case GPUDataType::UINT8:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::UINT8_1);
			break;
		case GPUDataType::UINT16:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::UINT16_1);
			break;
		case GPUDataType::UINT32:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::UINT32_1);
			break;
		case GPUDataType::FLOAT:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::FLOAT_1);
			break;
		case GPUDataType::DOUBLE:
			dataTypeStart = static_cast<uint32_t>(GFGDataType::DOUBLE_1);
			break;
	}
	
	sameType |= (static_cast<uint32_t>(gfg) == dataTypeStart + typeCount);
	return sameType;
}

void VertexBuffer::GenerateVertexBuffer()
{
	// Gen VAO
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// Bind Index Buffer to VAO
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.getGLBuffer());

	// Vertex
	glBindVertexBuffer(0, vertexBuffer.getGLBuffer(), 0,
					   static_cast<GLsizei>(byteStride));
	for(const VertexElement& currentElement : vElements)
	{
		glEnableVertexAttribArray(currentElement.inputPosition);

		if((currentElement.type != GPUDataType::FLOAT &&
			currentElement.type != GPUDataType::DOUBLE) &&
		   !currentElement.isNormInt)
		{
			glVertexAttribIFormat(currentElement.inputPosition,
								  currentElement.typeCount,
								  static_cast<GLenum>(currentElement.type),
								  static_cast<GLuint>(currentElement.offset));
		}
		else
		{
			glVertexAttribFormat(currentElement.inputPosition,
								 currentElement.typeCount,
								 static_cast<GLenum>(currentElement.type),
								 (currentElement.isNormInt) ? GL_TRUE : GL_FALSE,
								 static_cast<GLuint>(currentElement.offset));
		}
		glVertexAttribBinding(currentElement.inputPosition, 0);
	}
	GI_LOG("GPU\tVertex Definition Created. VAO ID: %d", vao);
}

void VertexBuffer::AddMesh(DrawPointIndexed& result,
						   const uint8_t data[],
						   const uint8_t indexData[],
						   size_t vertexCount,
						   size_t indexCount)
{
	assert(!addLocked);
	if(addLocked) return;

	// DP Populate
	result.baseInstance = static_cast<uint32_t>(meshOffsets.size());
	result.baseVertex = static_cast<uint32_t>(vertexBuffer.CPUData().size() / byteStride);
	result.firstIndex = static_cast<uint32_t>(indexBuffer.CPUData().size() / sizeof(uint32_t));
	result.count = static_cast<uint32_t>(indexCount);	
	result.instanceCount = 1;

	// Mesh start offset
	meshOffsets.push_back(result.baseVertex);

	// Acutal data
	vertexBuffer.CPUData().insert(vertexBuffer.CPUData().end(), data,
								  data + vertexCount * byteStride);
	indexBuffer.CPUData().insert(indexBuffer.CPUData().end(), indexData,
								 indexData + indexCount * sizeof(uint32_t));
}

void VertexBuffer::EditMesh(const uint8_t data[],
							uint32_t meshId,
							size_t vertexCount)
{
	size_t byteOffset = meshOffsets[meshId] * byteStride;
	size_t byteCount = vertexCount * byteStride;
	
	std::copy(data, data + byteCount, vertexBuffer.CPUData().begin() + byteOffset);
	if(addLocked) vertexBuffer.SendSubData(static_cast<uint32_t>(byteOffset), 
										   static_cast<uint32_t>(byteCount));
}

bool VertexBuffer::IsSuitedGFGMesh(const GFGMeshHeader& meshHeader)
{
	bool result = true;
	for(const GFGVertexComponent& comp : meshHeader.components)
	{
		for(const VertexElement& vElement : vElements)
		{
			result &= GFGSupportedLogic(comp.logic);
			if(GFGLogicHit(comp.logic, vElement.logic))
			{
				result &= GFGSameDataType(comp.dataType, vElement.type, vElement.typeCount);
				result &= comp.internalOffset == vElement.offset;
				result &= (comp.stride == 0) || (comp.stride == byteStride);
				result &= comp.startOffset == 0;
			}
		}
	}
	return result;
}

void VertexBuffer::Bind()
{
	assert(addLocked && vao != 0);
	glBindVertexArray(vao);
}

void VertexBuffer::LockAndLoad()
{
	// Lock Buffer for Adding Mesh and Send Data to GPU
	addLocked = true;

	vertexBuffer.SendData();
	indexBuffer.SendData();

	GenerateVertexBuffer();
}

void VertexBuffer::AttachMTransformIndexBuffer(GLuint transformIndexBuffer,
											   size_t transformIndexOffset)
{
	glBindVertexArray(vao);
	glBindVertexBuffer(1, 
					   transformIndexBuffer, 
					   transformIndexOffset, 
					   sizeof(uint32_t));
	glVertexBindingDivisor(1, 1);
	glEnableVertexAttribArray(IN_TRANS_INDEX);
	glVertexAttribIFormat(IN_TRANS_INDEX, 1, GL_UNSIGNED_INT, 0);
	glVertexAttribBinding(IN_TRANS_INDEX, 1);
}