#include "ConeTraceTexture.h"
#include <GLHeader.h>
#include "Macros.h"
#include "GLSLBindPoints.h"
#include "Camera.h"
#include "OGLTimer.h"

ConeTraceTexture::ConeTraceTexture()
	: frontTexture(0)
	, backTexture(0)
	, linearSampler(0)
	, width(0)
	, height(0)
	, format(0)
{}

ConeTraceTexture::ConeTraceTexture(GLsizei width, GLsizei height,
								   GLenum textureFormat)
	: frontTexture(0)
	, backTexture(0)
	, linearSampler(0)
	, compGauss16x16(ShaderType::COMPUTE, "Shaders/GaussBlur16x16.comp")
	, width(width)
	, height(height)
	, format(textureFormat)
{
	glGenTextures(1, &frontTexture);
	glBindTexture(GL_TEXTURE_2D, frontTexture);
	glTexStorage2D(GL_TEXTURE_2D, 1, textureFormat, width, height);

	glGenTextures(1, &backTexture);
	glBindTexture(GL_TEXTURE_2D, backTexture);
	glTexStorage2D(GL_TEXTURE_2D, 1, textureFormat, width, height);

	glGenSamplers(1, &linearSampler);
	glSamplerParameteri(linearSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(linearSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glSamplerParameteri(linearSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glSamplerParameteri(linearSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

ConeTraceTexture::ConeTraceTexture(ConeTraceTexture&& other)
	: frontTexture(other.frontTexture)
	, backTexture(other.backTexture)
	, linearSampler(other.linearSampler)
	, compGauss16x16(std::move(other.compGauss16x16))
	, width(other.width)
	, height(other.height)
	, format(other.format)
{
	other.frontTexture = 0;
	other.backTexture = 0;
	other.linearSampler = 0;
}

ConeTraceTexture& ConeTraceTexture::operator=(ConeTraceTexture&& other)
{
	assert(this != &other);

	glDeleteTextures(1, &frontTexture);
	glDeleteTextures(1, &backTexture);
	glDeleteSamplers(1, &linearSampler);
	
	frontTexture = other.frontTexture;
	backTexture = other.backTexture;
	linearSampler = other.linearSampler;
	compGauss16x16 = std::move(other.compGauss16x16);
	width = other.width;
	height = other.height;
	format = other.format;

	other.frontTexture = 0;
	other.backTexture = 0;
	other.linearSampler = 0;
	return *this;
}

ConeTraceTexture::~ConeTraceTexture()
{
	glDeleteTextures(1, &frontTexture);
	glDeleteTextures(1, &backTexture);
	glDeleteSamplers(1, &linearSampler);
}

GLuint ConeTraceTexture::Texture()
{
	return frontTexture;
}

double ConeTraceTexture::BlurTexture(GLuint depthBuffer, const Camera& camera)
{	
	// Timer
	OGLTimer t;
	t.Start();

	// Call Size
	GLuint blockX = 16;
	GLuint blockY = 16;
	GLuint gridX = (width + blockX - 1) / blockX;
	GLuint gridY = (height + blockY - 1) / blockY;
	
	// Edge Aware Gauss
	compGauss16x16.Bind();

	// Uniforms
	glUniform2f(U_NEAR_FAR, camera.near, camera.far);

	// Textures
	glActiveTexture(GL_TEXTURE0 + T_DEPTH);
	glBindTexture(GL_TEXTURE_2D, depthBuffer);	
	glBindSampler(T_DEPTH, linearSampler);
	
	//GLuint inTex = frontTexture;
	//GLuint outTex = backTexture;
	for(unsigned int i = 0; i < 1; i++)
	{
		// Call #1 (Vertical)
	    glActiveTexture(GL_TEXTURE0 + T_IN);
	    glBindTexture(GL_TEXTURE_2D, frontTexture);
	    glBindSampler(T_IN, linearSampler);
	    glBindImageTexture(I_OUT_TEXTURE, backTexture, 0, false, 0, GL_WRITE_ONLY, format);
	    glUniform1ui(U_DIRECTION, 0);
	    glDispatchCompute(gridX, gridY, 1);
	    
		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	    // Call #2 (Horizontal)
	    glActiveTexture(GL_TEXTURE0 + T_IN);
	    glBindTexture(GL_TEXTURE_2D, backTexture);
	    glBindSampler(T_IN, linearSampler);
	    glBindImageTexture(I_OUT_TEXTURE, frontTexture, 0, false, 0, GL_WRITE_ONLY, format);
	    glUniform1ui(U_DIRECTION, 1);
		glDispatchCompute(gridX, gridY, 1);

		glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	}

	t.Stop();
	return t.ElapsedMS();
}

GLsizei ConeTraceTexture::Width() const
{
	return width;
}

GLsizei ConeTraceTexture::Height() const
{
	return height;
}

GLenum ConeTraceTexture::Format() const
{
	return format;
}