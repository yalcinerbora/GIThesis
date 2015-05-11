/**

Structured buffer
Tempalte class that has both GPU and cpu image of the data
whgen data chagnes it resend the data bind time

*/

#ifndef __STRUCTUREDBUFFER_H__
#define __STRUCTUREDBUFFER_H__

#include "GLHeaderLite.h"
#include <vector>

template <class T>
class StructuredBuffer
{
	private:
		static	size_t			resizeFactor;

		GLuint					bufferId;
		size_t					bufferCapacity;
		std::vector<T>			dataGPUImage;
		bool					dataChanged;

		void					ResendData();

	protected:

	public:
								StructuredBuffer(size_t initialCapacity);
								StructuredBuffer(const StructuredBuffer&) = delete;
		const StructuredBuffer&	operator= (const StructuredBuffer&) = delete;
								~StructuredBuffer();

		void					AddData(const T&);
		GLuint					getGLBuffer();
		size_t					Count() const;

		void					BindAsUniformBuffer(GLuint location, 
													GLuint countOffset,
													GLuint countSize);
		void					BindAsUniformBuffer(GLuint location);
		void					BindAsShaderStorageBuffer(GLuint location, 
														  GLuint countOffset, 
														  GLuint countSize);
		void					BindAsShaderStorageBuffer(GLuint location);
		void					BindAsDrawIndirectBuffer();

		void					Resize(size_t count);
		void					SyncData(size_t newSize);
		T						GetData(uint32_t index);

		const std::vector<T>&	CPUData() const;
};
#include "StructuredBuffer.hpp"
#endif //__SCENE_H__
