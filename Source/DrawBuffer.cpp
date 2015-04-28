#include "DrawBuffer.h"
#include "GPUBuffer.h"
#include "IEUtility/IEMatrix4x4.h"
#include "Globals.h"

uint32_t DrawBuffer::drawParamSize = 512;
uint32_t DrawBuffer::drawParamFactor = 2;
uint32_t DrawBuffer::transformSize = 512;
uint32_t DrawBuffer::transformFactor = 2; 

DrawBuffer::DrawBuffer()
	: drawParamBuffer(0)
	, transformBuffer(0)
	, transSize(drawParamSize)
	, dpSize(transformSize)
{
	GLuint buffs[] = {0, 0};
	glGenBuffers(2, buffs);
	drawParamBuffer = buffs[0];
	transformBuffer = buffs[1];

	// User Non Pipeline Related Buffer Binding Poisition
	glBindBuffer(GL_COPY_WRITE_BUFFER, drawParamBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, drawParamSize * sizeof(DrawPointIndexed), nullptr,
				 GL_DYNAMIC_DRAW);

	glBindBuffer(GL_COPY_WRITE_BUFFER, transformBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, drawParamSize * sizeof(ModelTransform), nullptr,
					GL_DYNAMIC_DRAW);
}

DrawBuffer::~DrawBuffer()
{
	glDeleteBuffers(1, &transformBuffer);
	glDeleteBuffers(1, &drawParamBuffer);
}

// 
void DrawBuffer::AddMaterial(ColorMaterial c)
{
	materials.emplace_back(c);
}

void DrawBuffer::AddDrawCall(DrawPointIndexed dp,
							 uint32_t mIndex,
							 ModelTransform modelTransform)
{
	dataChanged = true;
	transformData.push_back(modelTransform);
	drawData.push_back(dp);
	materialIndex.push_back(mIndex);
}

void DrawBuffer::Draw()
{
	if(dataChanged)
	{
		if(drawData.size() > dpSize)
		{
			GLuint newBuffer;
			glGenBuffers(1, &newBuffer);

			glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
			glBufferData(GL_COPY_WRITE_BUFFER, dpSize * drawParamFactor,
							nullptr,
							GL_DYNAMIC_DRAW);

			glDeleteBuffers(1, &drawParamBuffer);
			drawParamBuffer = newBuffer;
			dpSize *= drawParamFactor;
		}
		if(transformData.size() > transSize)
		{
			GLuint newBuffer;
			glGenBuffers(1, &newBuffer);

			glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
			glBufferData(GL_COPY_WRITE_BUFFER, transSize * transformFactor,
							nullptr,
							GL_DYNAMIC_DRAW);

			glDeleteBuffers(1, &transformBuffer);
			transformBuffer = newBuffer;
			transSize *= transformFactor;
		}

		glBindBuffer(GL_COPY_WRITE_BUFFER, drawParamBuffer);
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0,
						drawData.size() * sizeof(DrawPointIndexed),
						drawData.data());

		glBindBuffer(GL_COPY_WRITE_BUFFER, transformBuffer);
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0,
						transformData.size() * sizeof(ModelTransform),
						transformData.data());

		dataChanged = false;
	}

	// Data is up to date now
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, drawParamBuffer);
	for(unsigned int i = 0; i < transformData.size(); i++)
	{
		materials[materialIndex[i]].BindMaterial();
		glBindBufferRange(GL_UNIFORM_BUFFER, U_MODEL_TRANSFORMS, transformBuffer,
						  i * sizeof(ModelTransform),
						  sizeof(ModelTransform));
		glDrawElementsIndirect(GL_TRIANGLES,
							   GL_UNSIGNED_INT,
							   (void *) (i * sizeof(DrawPointIndexed)));
	}
}