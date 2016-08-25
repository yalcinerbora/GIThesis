/**


*/

#ifndef __VOXFRAMEBUFFER_H__
#define __VOXFRAMEBUFFER_H__

#include <array>
#include "GLHeaderLite.h"

class VoxFramebuffer
{
	private:
		std::array<GLuint, 2>		renderBuffers;
		GLuint						fbo;

	protected:
	public:
									VoxFramebuffer(GLsizei width, GLsizei height);
									VoxFramebuffer(const VoxFramebuffer&) = delete;
		const VoxFramebuffer&		operator=(const VoxFramebuffer&) = delete;
									~VoxFramebuffer();

		void						Bind();
};
#endif //__VOXFRAMEBUFFER_H__