#pragma once

#include <GLHeaderLite.h>
#include "Shader.h"

struct Camera;

class ConeTraceTexture
{
	private:
		GLuint frontTexture;
		GLuint backTexture;
		GLuint linearSampler;

		Shader compGauss16x16;

		GLsizei width;
		GLsizei height;

		GLenum format;

	protected:
	public:
		// Constructors & Destructor
							ConeTraceTexture();
							ConeTraceTexture(GLsizei width, GLsizei height,
											 GLenum textureFormat);
							ConeTraceTexture(const ConeTraceTexture&) = delete;
							ConeTraceTexture(ConeTraceTexture&&);		
		ConeTraceTexture&	operator=(const ConeTraceTexture&) = delete;
		ConeTraceTexture&	operator=(ConeTraceTexture&&);
							~ConeTraceTexture();

		GLuint				Texture();
		void				BindAsTexture(GLuint target);
		double				BlurTexture(GLuint depthBuffer,
										const Camera& camera);

		GLsizei				Width() const;
		GLsizei				Height() const;
		GLenum				Format() const;
};
