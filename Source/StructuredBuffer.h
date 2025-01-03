/**

Structured buffer
Tempalte class that has both GPU and cpu image of the data
whgen data chagnes it resend the data bind time

*/

#ifndef __STRUCTUREDBUFFER_H__
#define __STRUCTUREDBUFFER_H__

#include "GLHeader.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <algorithm>

template <class T>
class StructuredBuffer
{
	private:
		static	size_t			resizeFactor;

		GLuint					bufferId;
		size_t					bufferCapacity;
		std::vector<T>			dataGPUImage;

		void					CheckBufferSize();

	protected:

	public:
		// Constructors & Destructor
								StructuredBuffer(size_t initialCapacity = 0, bool allocCPU = true);
								StructuredBuffer(StructuredBuffer&&);
								StructuredBuffer(const StructuredBuffer&) = delete;
		StructuredBuffer&		operator=(StructuredBuffer&&);
		StructuredBuffer&		operator=(const StructuredBuffer&) = delete;
								~StructuredBuffer();

		void					AddData(const T&);
		GLuint					getGLBuffer();
		size_t					Count() const;
		size_t					Capacity() const;

		void					Memset(uint8_t);
		void					Memset(uint32_t);
		void					Memset(uint32_t word, uint32_t offset, uint32_t size);
		void					BindAsUniformBuffer(GLuint location, 
													GLuint countOffset,
													GLuint countSize) const;
		void					BindAsUniformBuffer(GLuint location) const;
		void					BindAsShaderStorageBuffer(GLuint location, 
														  GLuint countOffset, 
														  GLuint countSize);
		void					BindAsShaderStorageBuffer(GLuint location);
		void					BindAsDrawIndirectBuffer();

		void					Resize(size_t count, bool resizeCPU = true);
		void					RecieveData(size_t newSize);
		T						GetData(uint32_t index);
		void					ChangeData(uint32_t index, const T& newData);
		void					SendData();
		void					SendSubData(uint32_t offset, uint32_t size);
		void					SendSubData(const T* data, uint32_t offset,
											uint32_t size);

		std::vector<T>&			CPUData();
		const std::vector<T>&	CPUData() const;
};
#include "StructuredBuffer.hpp"
#endif //__STRUCTUREDBUFFER_H__
