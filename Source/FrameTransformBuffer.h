/**

*/

#ifndef __FRAMETRANSFORM_H__
#define __FRAMETRANSFORM_H__

#include <cstdint>
#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"
#include "IEUtility/IEVector4.h"
#include "GLHeader.h"

#pragma pack(push, 1)
struct FrameTransformData
{
	IEMatrix4x4 view;
	IEMatrix4x4 projection;
};
#pragma pack(pop)

class FrameTransformBuffer
{
	private:
		GLuint		bufferId;

	protected:
	public:
		// Constructors & Destructor
								FrameTransformBuffer();
								FrameTransformBuffer(const FrameTransformBuffer&) = delete;
		FrameTransformBuffer&	operator=(const FrameTransformBuffer&) = delete;
								~FrameTransformBuffer();

		void					Update(FrameTransformBufferData);
		void					Bind();


};
#endif //__FRAMETRANSFORM_H__

