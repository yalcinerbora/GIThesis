/**

G-Buffer
It has minimal optimizations in order to save
memory bandwidth

All mats are using phong shading

All RT are 32 bit each
Layout:

	RT0 -> RGB Color, A specPower
	RT1 -> Normal (16X-15Y, MSB is the direction of Z)
	DS -> Depth Stencil (to get world position)

*/

#ifndef __GBUFFER_H__
#define __GBUFFER_H__

#include <cstdint>
#include "GLHeaderLite.h"

enum class RenderTargetLocation
{
	COLOR,
	NORMAL,
	DEPTH
};

class GBuffer
{
	private:
		GLuint		rtTextures[3];
		GLuint		fboTexSampler;
		GLuint		fbo;

		uint32_t    width;
		uint32_t    height;

	protected:
	public:
					GBuffer(GLuint w, GLuint h);
					~GBuffer();

		void		BindAsTexture(GLuint texTarget, 
								  RenderTargetLocation) const;
		void		BindAsFBO();
		void		AlignViewport();

		GLuint		getColorGL();
		GLuint		getDepthGL();
		GLuint		getNormalGL();

		static void	BindDefaultFBO();
};
#endif //__GBUFFER_H__