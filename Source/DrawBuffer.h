/**

Draw Buffer

Holds Transformation Matrices,
Holds draw point buffer

*/

#ifndef __DRAWBUFFER_H__
#define __DRAWBUFFER_H__

#include <vector>
#include "GLHeader.h"

struct DrawPointIndexed;
class IEMatrix4x4;

class DrawBuffer
{
	private:
		GLuint							drawParamBuffer;
		GLuint							transformBuffer;
		bool							dataEdited;

		std::vector<DrawPointIndexed>	drawData;
		std::vector<IEMatrix4x4>		transformData;
		std::vector<Material>			material;

	protected:
	public:
		// Constructors & Destructor
										DrawBuffer();
										DrawBuffer(const DrawBuffer&) = delete;
		const DrawBuffer&				operator=(const DrawBuffer&) = delete;
										~DrawBuffer();

		// 
		void							AddDrawCall(DrawPointIndexed, 
													const Material& material,
													IEMatrix4x4);
		void							Draw();

};
#endif //__DRAWBUFFER_H__