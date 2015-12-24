/**

SVO Tex Pool Used in svo tree to store

color opacity and normal


*/

#ifndef __SVOTEXPOOL_H__
#define __SVOTEXPOOL_H__

#include <array>
#include "GLHeaderLite.h"

#define MAX_TEX_ARRAY_COUNT 8
#define NODE_STORAGE_DIM_XY 5	// 5x5 blocks edge valued

class SVOTexPool
{
	private:
		static const GLsizei pageWidth = 1000;
		static const GLsizei pageHeight = 1000;
		static const GLsizei pageDepth = 128;
	
		std::array<GLuint, MAX_TEX_ARRAY_COUNT>	sparseTexArrays;
		

	protected:



	public:
		SVOTexPool()

};


#endif //__SVOTEXPOOL_H__
