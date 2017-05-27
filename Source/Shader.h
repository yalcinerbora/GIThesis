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
	COMPUTE,
	GEOMETRY
	// TODO: Add Tesseletion If necessary
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
		
		bool				GenSeperableProgam(const char fileName[], bool spirv,
											   const GLchar entryPointName[]);

	protected:

	public:
		// Constructors & Destructor
							Shader();
							Shader(ShaderType, const char fileName[], bool spirv = false,
								   const GLchar entryPointName[] = "main");
							Shader(const Shader&) = delete;
							Shader(Shader&&);
		Shader&				operator=(Shader&&);
		Shader&				operator=(const Shader&) = delete;
							~Shader();
		
		// Renderer Usage
		void				Bind();
		bool				IsValid() const;

		static void			Unbind(ShaderType);
};

#endif //__SHADER_H__