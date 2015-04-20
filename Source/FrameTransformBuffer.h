/**

*/

#ifndef __FRAMETRANSFORM_H__
#define __FRAMETRANSFORM_H__

#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"
#include "GLHeader.h"

#pragma pack(push, 1)
struct FrameTransformBufferData
{
	IEMatrix4x4 view;
	IEMatrix4x4 projection;
	IEMatrix3x3 rotation;
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

