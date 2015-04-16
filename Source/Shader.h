/**

Shader Class that Compiles and Binds Shaders

*/

#ifndef __SHADER_H__
#define __SHADER_H__

#include "GLHeaderLite.h"

enum class ShaderType
{
	VERTEX,
	FRAGMENT,
	COMPUTE
	// TODO: Add Tesseletion and Geometry If necessary
};

class Shader
{
	private:
		// Global Variables
		static GLuint		shaderPipelineID;

		// Properties 
		GLuint				shaderID;
		ShaderType			shaderType;
		bool				valid;

		static GLenum		ShaderTypeToGL(ShaderType);
		static GLenum		ShaderTypeToGLBit(ShaderType);
				
	protected:

	public:
		// Constructors & Destructor
							Shader(ShaderType, const char source[]);
							Shader(const Shader&) = delete;
		const Shader&		operator=(const Shader&) = delete;
							~Shader();
		
		// Renderer Usage
		void				Bind();
		bool				IsValid() const;

};

#endif //__SHADER_H__