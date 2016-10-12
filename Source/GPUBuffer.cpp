#include "GPUBuffer.h"
#include "Macros.h"
#include "Globals.h"
#include "GFG/GFGMeshHeader.h"

// 2 Megs of Buffer
uint32_t GPUBuffer::totalVertexCount = 8 * 1024 * 1024;
uint32_t GPUBuffer::totalIndexCount = 16 * 1024 * 1024;

GPUBuffer::GPUBuffer(const Array32<const VertexElement> elements)
	: usedVertexAmount(0)
	, usedIndexAmount(0)
	, vao(0)
	, vertexBuffer(0)
	, indexBuffer(0)
	, meshCount(0)
{
	// Gen VAO
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// GL Creation of Vertex Definition
	glGenBuffers(1, &indexBuffer);
	glGenBuffers(1, &vertexBuffer);

	// TODO: Hand Edited Part
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, totalIndexCount * sizeof(uint32_t), nullptr,
				 GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, totalVertexCount * sizeof(float) * 8, nullptr,
				 GL_DYNAMIC_DRAW);

	glBindVertexBuffer(0,
					   vertexBuffer,
					   0,
					   static_cast<GLintptr>(elements.arr[0].stride));

	for(unsigned int i = 0; i < elements.length; i++)
	{
		const VertexElement& current = elements.arr[i];
		glEnableVertexAttribArray(current.inputPosition);

		if((current.type != GPUDataType::FLOAT &&
			current.type != GPUDataType::DOUBLE) &&
			!current.isNormInt)
		{
			glVertexAttribIFormat(current.inputPosition,
								  current.typeCount,
								  static_cast<GLenum>(current.type),
								  static_cast<GLuint>(current.offset));
		}
		else
		{
			glVertexAttribFormat(current.inputPosition,
								 current.typeCount,
								 static_cast<GLenum>(current.type),
								 (current.isNormInt) ? GL_TRUE : GL_FALSE,
								 static_cast<GLuint>(current.offset));
		}
		glVertexAttribBinding(current.inputPosition, 0);
		vElements.insert(vElements.end(), elements.arr[i]);
	}
	GI_LOG("GPU\tVertex Definition Created. VAO ID: %d", vao);
}

GPUBuffer::~GPUBuffer()
{
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &indexBuffer);
	glDeleteVertexArrays(1, &vao);
}

bool GPUBuffer::AddMesh(DrawPointIndexed& result,
						const uint8_t data[],
						const uint8_t indexData[],
						size_t vertexStride,
						size_t vertexCount,
						size_t indexCount)
{
	result.baseInstance = meshCount;
	result.baseVertex = usedVertexAmount;
	result.count = static_cast<uint32_t>(indexCount);
	result.firstIndex = usedIndexAmount;
	result.instanceCount = 1;

	meshCount++;

	if(HasEnoughSpaceFor(vertexCount, indexCount))
	{
		glBindBuffer(GL_COPY_WRITE_BUFFER, vertexBuffer);
		glBufferSubData(GL_COPY_WRITE_BUFFER, 
						usedVertexAmount * vertexStride,
						vertexCount * vertexStride,
						data);
		usedVertexAmount += static_cast<uint32_t>(vertexCount);

		glBindBuffer(GL_COPY_WRITE_BUFFER, indexBuffer);
		glBufferSubData(GL_COPY_WRITE_BUFFER, 
						usedIndexAmount * sizeof(uint32_t),
						indexCount * sizeof(uint32_t),
						indexData);
		usedIndexAmount += static_cast<uint32_t>(indexCount);
		return true;
	}
	else
		return false;
}

bool GPUBuffer::IsSuitedGFGMesh(const GFGMeshHeader& meshHeader)
{
	bool result = true, hasPos = false, hasUV = false, hasNormal = false;
	for(const GFGVertexComponent& comp : meshHeader.components)
	{
		if(comp.logic == GFGVertexComponentLogic::POSITION)
		{
			hasPos = true;
			result &= comp.dataType == GFGDataType::FLOAT_3;
			result &= comp.internalOffset == 0;
			result &= (comp.stride == 0) || (comp.stride == sizeof(float) * 8);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::UV)
		{
			hasUV = true;
			result &= comp.dataType == GFGDataType::FLOAT_2;
			result &= comp.internalOffset == 6 * sizeof(float);
			result &= (comp.stride == 0) || (comp.stride == sizeof(float) * 8);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::NORMAL)
		{
			hasNormal = true;
			result &= comp.dataType == GFGDataType::FLOAT_3;
			result &= comp.internalOffset == 3 * sizeof(float);
			result &= (comp.stride == 0) || (comp.stride == sizeof(float) * 8);
			result &= comp.startOffset == 0;
		}
		else
		{
			result &= false;
		}
	}
	return result && hasPos && hasUV && hasNormal;
}

bool GPUBuffer::IsSuitedGFGMeshSkeletal(const GFGMeshHeader& meshHeader)
{
	bool result = true, hasPos = false, hasUV = false, hasNormal = false;
	bool hasWeight = false, hasWIndex = false;
	for(const GFGVertexComponent& comp : meshHeader.components)
	{
		uint64_t stride = sizeof(float) * 8 + sizeof(uint8_t) * 8;
		if(comp.logic == GFGVertexComponentLogic::POSITION)
		{
			hasPos = true;
			result &= comp.dataType == GFGDataType::FLOAT_3;
			result &= comp.internalOffset == 0;
			result &= (comp.stride == 0) || (comp.stride == stride);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::UV)
		{
			hasUV = true;
			result &= comp.dataType == GFGDataType::FLOAT_2;
			result &= comp.internalOffset == 6 * sizeof(float);
			result &= (comp.stride == 0) || (comp.stride == stride);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::NORMAL)
		{
			hasNormal = true;
			result &= comp.dataType == GFGDataType::FLOAT_3;
			result &= comp.internalOffset == 3 * sizeof(float);
			result &= (comp.stride == 0) || (comp.stride == stride);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::WEIGHT)
		{
			hasWeight = true;
			result &= comp.dataType == GFGDataType::UNORM8_4;
			result &= comp.internalOffset == 8 * sizeof(float);
			result &= (comp.stride == 0) || (comp.stride == stride);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::WEIGHT_INDEX)
		{
			hasWIndex = true;
			result &= comp.dataType == GFGDataType::UINT8_4;
			result &= comp.internalOffset == 8 * sizeof(float) + 4 * sizeof(uint8_t);
			result &= (comp.stride == 0) || (comp.stride == stride);
			result &= comp.startOffset == 0;
		}
		else
		{
			result &= false;
		}
	}
	return result && hasPos && hasUV && hasNormal && hasWeight && hasWIndex;
}

bool GPUBuffer::HasEnoughSpaceFor(uint64_t vertexCount,
								  uint64_t indexCount)
{
	return	(vertexCount <= totalVertexCount - usedVertexAmount) &&
			(indexCount <= totalIndexCount - usedIndexAmount);
}

void GPUBuffer::Bind()
{
	glBindVertexArray(vao);
}

void GPUBuffer::AttachMTransformIndexBuffer(GLuint transformIndexBuffer)
{
	glBindVertexArray(vao);
	glBindVertexBuffer(1,
					   transformIndexBuffer,
					   0,
					   sizeof(uint32_t));
	glVertexBindingDivisor(1, 1);
	glEnableVertexAttribArray(IN_TRANS_INDEX);
	glVertexAttribIFormat(IN_TRANS_INDEX, 1, GL_UNSIGNED_INT, 0);
	glVertexAttribBinding(IN_TRANS_INDEX, 1);
}