/**

Draw Buffer

Holds Transformation Matrices,
Holds draw point buffer

*/

#ifndef __DRAWBUFFER_H__
#define __DRAWBUFFER_H__

#include <vector>
#include "GLHeader.h"
#include "Material.h"
#include "IEUtility/IEMatrix4x4.h"
#include "IEUtility/IEMatrix3x3.h"

struct DrawPointIndexed;

struct ModelTransform
{
	IEMatrix4x4 model;
	IEMatrix3x3 modelRotation;

	// TODO: This is a bullshit solution it only works on my cards but w/e
	// OffsetAlignment
	uint8_t offset[256 - sizeof(IEMatrix4x4) - sizeof(IEMatrix3x3)];
};

class DrawBuffer
{
	private:
		static uint32_t					drawParamSize;
		static uint32_t					drawParamFactor;
		static uint32_t					transformFactor;
		static uint32_t					transformSize;

		GLuint							drawParamBuffer;
		GLuint							transformBuffer;
		uint32_t						transSize;
		uint32_t						dpSize;
		bool							dataChanged;

		std::vector<DrawPointIndexed>	drawData;
		std::vector<ModelTransform>		transformData;
		std::vector<uint32_t>			materialIndex;
		std::vector<Material>			materials;

	protected:
	public:
		// Constructors & Destructor
										DrawBuffer();
										DrawBuffer(const DrawBuffer&) = delete;
		const DrawBuffer&				operator=(const DrawBuffer&) = delete;
										~DrawBuffer();

		// 
		void							AddMaterial(ColorMaterial);
		void							AddDrawCall(DrawPointIndexed, 
													uint32_t materialIndex,
													ModelTransform modelTransform);
		void							Draw();
};
#endif //__DRAWBUFFER_H__