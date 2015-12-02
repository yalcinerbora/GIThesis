
template <class T>
size_t StructuredBuffer<T>::resizeFactor = 2;

template <class T>
StructuredBuffer<T>::StructuredBuffer(size_t initialCapacity)
	: bufferId(0)
	, bufferCapacity(initialCapacity)
{
	assert(initialCapacity != 0);
	glGenBuffers(1, &bufferId);
	glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
	glBufferData(GL_COPY_WRITE_BUFFER, bufferCapacity * sizeof(T),
				 nullptr, GL_DYNAMIC_DRAW);
}

template <class T>
StructuredBuffer<T>::StructuredBuffer(StructuredBuffer&& other)
	: bufferId(other.bufferId)
	, bufferCapacity(other.bufferCapacity)
	, dataGPUImage(std::move(other.dataGPUImage))
{
	other.bufferId = 0;
	other.bufferCapacity = 0;
}

template <class T>
StructuredBuffer<T>::~StructuredBuffer()
{
	glDeleteBuffers(1, &bufferId);
}

template <class T>
void StructuredBuffer<T>::SendData()
{
	if(dataGPUImage.size() > bufferCapacity)
	{
		bufferCapacity = bufferCapacity * resizeFactor;

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
					0,
					dataGPUImage.size() * sizeof(T),
					dataGPUImage.data());
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
size_t  StructuredBuffer<T>::Capacity() const
{
	return bufferCapacity;
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
void StructuredBuffer<T>::Resize(size_t count)
{
	if(count < bufferCapacity) return;

	GLuint newBuffer;
	glGenBuffers(1, &newBuffer);

	glBindBuffer(GL_COPY_WRITE_BUFFER, newBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, count * sizeof(T), 
				 nullptr,
				 GL_DYNAMIC_DRAW);

	glBindBuffer(GL_COPY_READ_BUFFER, bufferId);
	glCopyBufferSubData(GL_COPY_READ_BUFFER,
						GL_COPY_WRITE_BUFFER,
						0, 0,
						dataGPUImage.size() * sizeof(T));

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