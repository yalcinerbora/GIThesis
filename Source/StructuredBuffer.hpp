
template <class T>
float StructuredBuffer<T>::resizeFactor = 1.5f;

template <class T>
StructuredBuffer<T>::StructuredBuffer(size_t initialCapacity)
	: bufferId(0)
	, bufferCapacity(initialCapacity)
	, dataChanged(true)
{
	glGenBuffers(1, &bufferId);
	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glBufferData(GL_COPY_WRITE_BUFFER, bufferCapacity * sizeof(T),
				 nullptr, GL_DYNAMIC_DRAW);
}

template <class T>
StructuredBuffer<T>::~StructuredBuffer()
{
	glDeleteBuffers(1, &bufferId);
}

template <class T>
void  StructuredBuffer<T>::ResendData()
{
	if(dataChanged)
	{
		if(dataGPUImage.size() > bufferCapacity)
		{
			bufferCapacity = static_cast<size_t>(bufferCapacity * resizeFactor);

			GLuint newBuffer;

			// Param Buffer
			glGenBuffers(1, &newBuffer);
			glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
			glBufferData(GL_COPY_WRITE_BUFFER, bufferCapacity * sizeof(T),
						 nullptr,
						 GL_DYNAMIC_DRAW);
			glDeleteBuffers(1, &bufferId);
			bufferId = newBuffer;
		}

		glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
		glBufferSubData(GL_COPY_WRITE_BUFFER, 0,
						dataGPUImage.size() * sizeof(T),
						dataGPUImage.data());
		dataChanged = false;
	}
}

template <class T>
void StructuredBuffer<T>::AddData(const T& t)
{
	dataGPUImage.push_back(t);
	dataChanged = true;
}

template <class T>
GLuint StructuredBuffer<T>::getGLBuffer()
{
	ResendData();
	return bufferId;
}

template <class T>
size_t  StructuredBuffer<T>::Count() const
{
	return dataGPUImage.size();
}

template <class T>
void StructuredBuffer<T>::BindAsUniformBuffer(GLuint location,
											  GLuint countOffset,
											  GLuint countSize)
{
	ResendData();
	glBindBufferRange(GL_UNIFORM_BUFFER, location, bufferId,
					  countOffset * sizeof(T),
					  countSize * sizeof(T));
}

template <class T>
void StructuredBuffer<T>::BindAsUniformBuffer(GLuint location)
{
	ResendData();
	glBindBufferBase(GL_UNIFORM_BUFFER, location, bufferId);
}

template <class T>
void StructuredBuffer<T>::BindAsShaderStorageBuffer(GLuint location,
													GLuint countOffset,
													GLuint countSize)
{
	ResendData();
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, location,
					  bufferId,
					  countOffset * sizeof(T),
					  countSize * sizeof(T));
}

template <class T>
void StructuredBuffer<T>::BindAsShaderStorageBuffer(GLuint location)
{
	ResendData();
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, bufferId);

}

template <class T>
void StructuredBuffer<T>::BindAsDrawIndirectBuffer()
{
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufferId);
}