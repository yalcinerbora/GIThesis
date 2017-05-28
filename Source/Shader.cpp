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


bool Shader::GenSeperableProgam(const char fileName[], bool spirv,
								const GLchar entryPointName[])
{
	std::string file(fileName);
	size_t pos = file.find_last_of("\\/");
	pos = (pos == std::string::npos) ? 0 : (pos + 1);
	std::string onlyFileName = file.substr(pos);

	file += (spirv) ? (".spirv") : "";

	std::vector<GLchar> source;
	source.resize(std::ifstream(fileName, std::ifstream::ate | std::ifstream::binary).tellg());
	std::ifstream shaderFile(fileName);
	shaderFile.read(source.data(), source.size());

	const GLuint shader = glCreateShader(ShaderTypeToGL(shaderType));
	if(spirv)
	{
		// spir-v binary load
		glShaderBinary(1, &shader, GL_SHADER_BINARY_FORMAT_SPIR_V_ARB,
					   source.data(),
					   static_cast<GLsizei>(source.size()));
		glSpecializeShaderARB(shader,
							  entryPointName,
							  0,
							  nullptr,
							  nullptr);
	}
	else
	{
		const GLchar* sourcePtr = source.data();
		const GLint sourceSize = static_cast<GLint>(source.size());
		glShaderSource(shader, 1, &sourcePtr, &sourceSize);
		glCompileShader(shader);
	}
	
	GLint compiled = 0;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
	if(compiled == GL_FALSE)
	{
		GLint blen = 0;
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &blen);
		std::vector<GLchar> log(blen);
		glGetShaderInfoLog(shader, blen, &blen, &log[0]);
		GI_ERROR_LOG("Shader Compilation Error on File %s :\n%s", onlyFileName.c_str(), &log[0]);
		return false;
	}
	else
	{
		shaderID = glCreateProgram();
		glProgramParameteri(shaderID, GL_PROGRAM_SEPARABLE, GL_TRUE);
		glAttachShader(shaderID, shader);
		glLinkProgram(shaderID);
		glDetachShader(shaderID, shader);
		GI_LOG("Shader Compiled Successfully. Shader ID: %d, Name: %s", shaderID, onlyFileName.c_str());
	}
	glDeleteShader(shader);
	return true;
}

Shader::Shader()
	: valid(false)
	, shaderID(0)
	, shaderType(ShaderType::VERTEX)
{}

Shader::Shader(ShaderType t, const char fileName[], bool spirv,
			   const GLchar entryPointName[])
	: valid(false)
	, shaderID(0)
	, shaderType(t)
{
	// Create Pipeline If not Avail
	if(shaderPipelineID == 0)
	{
		glGenProgramPipelines(1, &shaderPipelineID);
		glBindProgramPipeline(shaderPipelineID);
	}
	valid = GenSeperableProgam(fileName, spirv, entryPointName);
}

Shader::Shader(Shader&& other)
	: shaderID(other.shaderID)
	, shaderType(other.shaderType)
	, valid(other.valid)
{
	other.shaderID = 0;
}

Shader& Shader::operator=(Shader&& other)
{
	assert(this != &other);
	shaderID = other.shaderID;
	shaderType = other.shaderType;
	valid = other.valid;
	other.shaderID = 0;
	return *this;
}

Shader::~Shader()
{
	glDeleteProgram(shaderID);
	if(shaderID !=  0) GI_LOG("Shader Deleted. Shader ID: %d", shaderID);
	shaderID = 0;

	// We Need To Delete Pipeline Somehow Here
	// Best way is to not delete and leave it be
	// Or use some RAII but needs to be deleted before context deletion
	// else it will not get deleted anyway

	// TODO: Dont Leak OGL ShaderProgram pipeline object
}

void Shader::Bind()
{
	assert(valid);
	glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), shaderID);
	glActiveShaderProgram(shaderPipelineID, shaderID);
}

bool Shader::IsValid() const
{
	return valid;
}

void Shader::Unbind(ShaderType shaderType)
{
	glUseProgramStages(shaderPipelineID, ShaderTypeToGLBit(shaderType), 0);
	glActiveShaderProgram(shaderPipelineID, 0);
}