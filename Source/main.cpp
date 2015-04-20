#include <iostream>
#include <GFG/GFGHeader.h>

#include "Window.h"

#include "NoInput.h"
#include "FPSInput.h"
#include "MayaInput.h"

#include "Globals.h"
#include "GFGLoader.h"

// GPU Data
#include "DrawBuffer.h"
#include "GPUBuffer.h"
#include "Shader.h"
#include "Camera.h"

#include "GLFW/glfw3.h"

int main()
{
	Camera mainRenderCamera =
	{
		80.0f,
		0.1f,
		10000.0f,
		1280,
		720,
		{ 600.0f, -400.0f, -4.5f },
		IEVector3::ZeroVector,
		IEVector3::Yaxis
	};

	// Input Schemes
	NoInput nullInput(mainRenderCamera);		// No Input from peripheral devices
	FPSInput fpsInput(mainRenderCamera);
	MayaInput mayaInput(mainRenderCamera);

	// Window Init
	WindowProperties winProps
	{
		1280,
		720,
		WindowScreenType::WINDOWED
	};
	Window mainWindow(nullInput,
					  winProps);


	// Vertex Element
	struct VAO
	{
		float vPos[3];
		float vNormal[3];
		float vUV[2];
	};
	VertexElement element[] = 
	{
		{
			0,
			GPUDataType::FLOAT,
			3,
			offsetof(struct VAO, vPos),
			sizeof(VAO)
		},
		{
			1,
			GPUDataType::FLOAT,
			3,
			offsetof(struct VAO, vNormal),
			sizeof(VAO)
		},
		{
			2,
			GPUDataType::FLOAT,
			2,
			offsetof(struct VAO, vUV),
			sizeof(VAO)
		}
	};
	GPUBuffer vertices({element, 3});
	DrawBuffer draw;
	FrameTransformBuffer cameraTransform;

	// Load GFG
	GFGLoader::LoadGFG(vertices, draw, "crySponza.gfg");

	// Load Shader
	Shader vertexGBufferWrite(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert");
	Shader fragmentGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag");

	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	// All Init
	// Render Loop
	while(!mainWindow.WindowClosed())
	{
		// Start With a VP Set
		// Using a callback is not necessarly true since it may alter some framebuffer's viewport
		// but we have to be sure that it alters main fbo viewport
		glViewport(0, 0, 
				   static_cast<GLsizei>(mainRenderCamera.width), 
				   static_cast<GLsizei>(mainRenderCamera.height));

		glClear(GL_COLOR_BUFFER_BIT |
				GL_DEPTH_BUFFER_BIT |
				GL_STENCIL_BUFFER_BIT);

		// Camera Transform
		cameraTransform.Update(mainRenderCamera.generateTransform());
		cameraTransform.Bind();

		// Shaders
		vertexGBufferWrite.Bind();
		fragmentGBufferWrite.Bind();

		// Vertices
		vertices.Bind();

		// Finally Draw Call
		draw.Draw();

		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();
	}
	return 0;
}