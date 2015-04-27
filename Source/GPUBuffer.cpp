#include "GPUBuffer.h"
#include "Macros.h"

// 2 Megs of Buffer
uint32_t GPUBuffer::totalVertexCount = 2 * 1024 * 1024;
uint32_t GPUBuffer::totalIndexCount = 4 * 1024 * 1024;

GPUBuffer::GPUBuffer(const Array32<const VertexElement> elements)
	: usedVertexAmount(0)
	, usedIndexAmount(0)
	, vao(0)
	, vertexBuffer(0)
	, indexBuffer(0)
{
	// Gen VAO
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	// GL Creation of Vertex Definition
	glGenBuffers(1, &indexBuffer);
	glGenBuffers(1, &vertexBuffer);

	// TODO: Hand Edited Part
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, totalVertexCount * sizeof(uint32_t), nullptr,
					GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, totalIndexCount * sizeof(float) * 8, nullptr,
				 GL_DYNAMIC_DRAW);

	for(unsigned int i = 0; i < elements.length; i++)
	{
		const VertexElement& current = elements.arr[i];
		glBindVertexBuffer(static_cast<GLuint>(i),
						   vertexBuffer,
						   0,
						   static_cast<GLintptr>(current.stride));
		glEnableVertexAttribArray(current.inputPosition);
		glVertexAttribFormat(current.inputPosition,
							 current.typeCount,
							 static_cast<GLenum>(current.type),
							 GL_FALSE,
							 static_cast<GLuint>(current.offset));
		glVertexAttribBinding(current.inputPosition, static_cast<GLuint>(i));
		vElements.insert(vElements.begin(), elements.arr[i]);
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
						size_t vertexCount,
						size_t indexCount)
{
	result.baseInstance = 0;
	result.baseVertex = usedVertexAmount;
	result.count = static_cast<uint32_t>(indexCount);
	result.firstIndex = usedIndexAmount;
	result.instanceCount = 1;

	if(HasEnoughSpaceFor(vertexCount, indexCount))
	{
		glBindBuffer(GL_COPY_WRITE_BUFFER, vertexBuffer);
		glBufferSubData(GL_COPY_WRITE_BUFFER, 
						usedVertexAmount * sizeof(float) * 8, 
						vertexCount * sizeof(float) * 8,
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
	bool result = true;
	for(const GFGVertexComponent& comp : meshHeader.components)
	{
		if(comp.logic == GFGVertexComponentLogic::POSITION)
		{
			result &= comp.dataType == GFGDataType::FLOAT_3;
			result &= comp.internalOffset == 0;
			result &= (comp.stride == 0) || (comp.stride == sizeof(float) * 8);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::UV)
		{
			result &= comp.dataType == GFGDataType::FLOAT_2;
			result &= comp.internalOffset == 6 * sizeof(float);
			result &= (comp.stride == 0) || (comp.stride == sizeof(float) * 8);
			result &= comp.startOffset == 0;
		}
		else if(comp.logic == GFGVertexComponentLogic::NORMAL)
		{
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
	return result;
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