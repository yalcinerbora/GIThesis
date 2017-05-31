
template <class T>
size_t StructuredBuffer<T>::resizeFactor = 2;

template <class T>
StructuredBuffer<T>::StructuredBuffer(size_t initialCapacity, bool allocCPU)
	: bufferId(0)
	, bufferCapacity(initialCapacity)
{
	if(bufferCapacity != 0)
	{
		if(allocCPU) dataGPUImage.resize(initialCapacity);
		glGenBuffers(1, &bufferId);
		glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
		glBufferData(GL_COPY_WRITE_BUFFER, bufferCapacity * sizeof(T),
					 nullptr, GL_DYNAMIC_DRAW);
	}
}

template <class T>
StructuredBuffer<T>::StructuredBuffer(StructuredBuffer&& other)
	: bufferId(other.bufferId)
	, bufferCapacity(other.bufferCapacity)
	, dataGPUImage(std::move(other.dataGPUImage))
{
	other.bufferId = 0;
}

template <class T>
StructuredBuffer<T>& StructuredBuffer<T>::operator=(StructuredBuffer&& other)
{
	assert(this != &other);
	bufferId = other.bufferId;
	bufferCapacity = other.bufferCapacity;
	dataGPUImage = std::move(other.dataGPUImage);

	other.bufferId = 0;
	return *this;
}

template <class T>
StructuredBuffer<T>::~StructuredBuffer()
{
	glDeleteBuffers(1, &bufferId);
}

template <class T>
void StructuredBuffer<T>::CheckBufferSize()
{
	if(dataGPUImage.size() > bufferCapacity)
	{
		bufferCapacity = std::max(dataGPUImage.size(), bufferCapacity * resizeFactor);
		if(bufferCapacity == 0) bufferCapacity = dataGPUImage.size();

		GLuint newBuffer;

		// Param Buffer
		glGenBuffers(1, &newBuffer);
		glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
		glBufferData(GL_COPY_WRITE_BUFFER,
					 bufferCapacity * sizeof(T),
					 nullptr,
					 GL_DYNAMIC_DRAW);
		glDeleteBuffers(1, &bufferId);
		bufferId = newBuffer;
	}
}

template <class T>
void StructuredBuffer<T>::SendData()
{
	if(dataGPUImage.size() > bufferCapacity)
	{
		bufferCapacity = std::max(dataGPUImage.size(), bufferCapacity * resizeFactor);
		if(bufferCapacity == 0) bufferCapacity = dataGPUImage.size();

		// Param Buffer
		GLuint newBuffer;
		glGenBuffers(1, &newBuffer);
		glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
		glBufferData(GL_COPY_WRITE_BUFFER,
					 bufferCapacity * sizeof(T),
					 nullptr,
					 GL_DYNAMIC_DRAW);
		glDeleteBuffers(1, &bufferId);
		bufferId = newBuffer;
	}

	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glBufferSubData(GL_COPY_WRITE_BUFFER,
					0,
					dataGPUImage.size() * sizeof(T),
					dataGPUImage.data());
}

template <class T>
void StructuredBuffer<T>::SendSubData(uint32_t offset, uint32_t size)
{
	if(dataGPUImage.size() > bufferCapacity)
	{
		bufferCapacity = std::max(dataGPUImage.size(), bufferCapacity * resizeFactor);
		if(bufferCapacity == 0) bufferCapacity = dataGPUImage.size();

		GLuint newBuffer;

		// Param Buffer
		glGenBuffers(1, &newBuffer);
		glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
		glBufferData(GL_COPY_WRITE_BUFFER,
					 bufferCapacity * sizeof(T),
					 nullptr,
					 GL_DYNAMIC_DRAW);
		glDeleteBuffers(1, &bufferId);
		bufferId = newBuffer;
	}

	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glBufferSubData(GL_COPY_WRITE_BUFFER,
					offset * sizeof(T),
					size * sizeof(T),
					dataGPUImage.data() + offset * sizeof(T));
}

template <class T>
void StructuredBuffer<T>::AddData(const T& t)
{
	dataGPUImage.push_back(t);
}

template <class T>
GLuint StructuredBuffer<T>::getGLBuffer()
{
	return bufferId;
}

template <class T>
size_t StructuredBuffer<T>::Count() const
{
	return dataGPUImage.size();
}

template <class T>
size_t StructuredBuffer<T>::Capacity() const
{
	return bufferCapacity;
}

template <class T>
void StructuredBuffer<T>::Memset(uint8_t byte)
{
	glInvalidateBufferData(bufferId);
	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R8UI, GL_RED,
					  GL_UNSIGNED_BYTE, &byte);
}

template <class T>
void StructuredBuffer<T>::Memset(uint32_t word)
{
	glInvalidateBufferData(bufferId);
	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);	
	glClearBufferData(GL_COPY_WRITE_BUFFER, GL_R32UI, GL_RED,
					  GL_UNSIGNED_INT, &word);
}

template <class T>
void StructuredBuffer<T>::BindAsUniformBuffer(GLuint location,
											  GLuint countOffset,
											  GLuint countSize)
{
	glBindBufferRange(GL_UNIFORM_BUFFER, location, bufferId,
					  countOffset * sizeof(T),
					  countSize * sizeof(T));
}

template <class T>
void StructuredBuffer<T>::BindAsUniformBuffer(GLuint location)
{
	glBindBufferBase(GL_UNIFORM_BUFFER, location, bufferId);
}

template <class T>
void StructuredBuffer<T>::BindAsShaderStorageBuffer(GLuint location,
													GLuint countOffset,
													GLuint countSize)
{
	glBindBufferRange(GL_SHADER_STORAGE_BUFFER, location,
					  bufferId,
					  countOffset * sizeof(T),
					  countSize * sizeof(T));
}

template <class T>
void StructuredBuffer<T>::BindAsShaderStorageBuffer(GLuint location)
{
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, location, bufferId);

}

template <class T>
void StructuredBuffer<T>::BindAsDrawIndirectBuffer()
{
	glBindBuffer(GL_DRAW_INDIRECT_BUFFER, bufferId);
}

template <class T>
void StructuredBuffer<T>::Resize(size_t count, bool resizeCPU)
{
	assert(count != 0);
	if(count < bufferCapacity) return;

	GLuint newBuffer;
	glGenBuffers(1, &newBuffer);

	glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, count * sizeof(T), 
				 nullptr,
				 GL_DYNAMIC_DRAW);

	if(bufferId != 0)
	{
		glBindBuffer(GL_COPY_READ_BUFFER, bufferId);
		glCopyBufferSubData(GL_COPY_READ_BUFFER,
							GL_COPY_WRITE_BUFFER,
							0, 0,
							dataGPUImage.size() * sizeof(T));
	}
	glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
	if(resizeCPU) dataGPUImage.resize(count);

	glDeleteBuffers(1, &bufferId);
	bufferId = newBuffer;
	bufferCapacity = count;
}

template <class T>
void StructuredBuffer<T>::RecieveData(size_t newSize)
{
	// Data Altered on the buffer
	// Move this data to CPU
	dataGPUImage.resize(newSize);
	glBindBuffer(GL_COPY_READ_BUFFER, bufferId);
	glGetBufferSubData(GL_COPY_READ_BUFFER, 0, newSize * sizeof(T),
					   dataGPUImage.data());
}

template <class T>
void StructuredBuffer<T>::ChangeData(uint32_t index, const T& newData)
{
	assert(index < dataGPUImage.size());
	dataGPUImage[index] = newData;
	glBindBuffer(GL_COPY_READ_BUFFER, bufferId);
	glBufferSubData(GL_COPY_READ_BUFFER, index * sizeof(T), sizeof(T),
					   dataGPUImage.data() + index);
}

template <class T>
std::vector<T>& StructuredBuffer<T>::CPUData()
{
	return dataGPUImage;
}

template <class T>
const std::vector<T>& StructuredBuffer<T>::CPUData() const
{
	return dataGPUImage;
}

template <class T>
T StructuredBuffer<T>::GetData(uint32_t index)
{
	assert(index < dataGPUImage.size());
	glBindBuffer(GL_COPY_READ_BUFFER, bufferId);
	glGetBufferSubData(GL_COPY_READ_BUFFER, index * sizeof(T), sizeof(T),
					   dataGPUImage.data() + index);
	return dataGPUImage[index];
}