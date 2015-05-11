#include "Shader.h"
#include "GLHeader.h"
#include "Macros.h"

#include <vector>
#include <fstream>

GLuint Shader::shaderPipelineID = 0;

GLenum Shader::ShaderTypeToGL(ShaderType t)
	
{
	static GLenum values[] = 
	{
		GL_VERTEX_SHADER,
		GL_FRAGMENT_SHADER,
		GL_COMPUTE_SHADER,
		GL_GEOMETRY_SHADER
	};
	return values[static_cast<int>(t)];
}

GLenum Shader::ShaderTypeToGLBit(ShaderType t)
{
	static GLenum values[] =
	{
		GL_VERTEX_SHADER_BIT,
		GL_FRAGMENT_SHADER_BIT,
		GL_COMPUTE_SHADER_BIT,
		GL_GEOMETRY_SHADER_BIT
	};
	return values[static_cast<int>(t)];
}

Shader::Shader(ShaderType t, const char fileName[])
	: valid(false)
	, shaderID(0)
	, shaderType(t)
{
	std::vector<char> source;
	source.resize(std::ifstream(fileName, std::ifstream::ate | std::ifstream::binary).tellg());
	std::ifstream shaderFile(fileName);
	shaderFile.read(source.data(), source.size());

	// Create Pipeline If not Avail
	if(shaderPipelineID == 0)
	{
		glGenProgramPipelines(1, &shaderPipelineID);
		glBindProgramPipeline(shaderPipelineID);
	}

	// Compile
	const char* sourcePtr = source.data();
	shaderID = glCreateShaderProgramv(ShaderTypeToGL(shaderType), 1, (const GLchar**) &sourcePtr);

	GLint result;
	glGetProgramiv(shaderID, GL_LINK_STATUS, &result);
	// Check Errors
	if(result == GL_FALSE)
	{
		GLint blen = 0;
		glGetProgramiv(shaderID, GL_INFO_LOG_LENGTH, &blen);
		if(blen > 1)
		{
			std::vector<GLchar> log(blen);
			glGetProgramInfoLog(shaderID, blen, &blen, &log[0]);
			GI_ERROR_LOG("Shader Compilation Error on File %s :\n%s", fileName, &log[0]);
		}
	}
	else
	{
		GI_LOG("Shader Compiled Successfully. Shader ID: %d", shaderID);
		valid = true;
	}

}

Shader::~Shader()
{
	glDeleteProgram(shaderID);
	GI_LOG("Shader Deleted. Shader ID: %d", shaderID);
	shaderID = 0;

	// We Need To Delete Pipeline Somehow Here
	// Best way is to not delete and leave it be
	// Or use some RAII but needs to be deleted before context deletion
	// else it will not get deleted anyway

	// TODO: Dont Leak OGL ShaderProgram pipeline object
}

void Shader::Bind()
{
	glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), shaderID);
	glActiveShaderProgram(shaderPipelineID, shaderID);
}

bool Shader::IsValid() const
{
	return valid;
}

void  Shader::Unbind(ShaderType shaderType)
{
	glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), 0);
	glActiveShaderProgram(shaderPipelineID, 0);
}