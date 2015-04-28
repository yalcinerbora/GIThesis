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
#include "Scene.h"

#include "GLFW/glfw3.h"

int main()
{
	Camera mainRenderCamera =
	{
		60.0f,
		0.1f,
		10000.0f,
		1280,
		720,
		{ 150.0f, -145.0f, 0.3f },
		{ -0.15f, -37.0f, 3.4f },
		IEVector3::Yaxis
	};

	uint32_t currentSolution = 0, currentScene = 0, currentInputScheme = 0;
	std::vector<SolutionI*>	solutions;
	std::vector<SceneI*>	scenes;
	std::vector<InputManI*>	inputSchemes;

	// Input Schemes
	NoInput nullInput(mainRenderCamera, currentSolution, currentScene, currentInputScheme);
	MayaInput mayaInput(mainRenderCamera, currentSolution, currentScene, currentInputScheme);
	FPSInput fpsInput(mainRenderCamera, currentSolution, currentScene, currentInputScheme);
	inputSchemes.push_back(&nullInput);
	inputSchemes.push_back(&mayaInput);
	inputSchemes.push_back(&fpsInput);

	// Window Init
	WindowProperties winProps
	{
		1280,
		720,
		WindowScreenType::WINDOWED
	};
	Window mainWindow(nullInput,
					  winProps);

	// Camera GPU
	FrameTransformBuffer cameraTransform;

	// Shaders
	Shader vertexGBufferWrite(ShaderType::VERTEX, "Shaders/GWriteGeneric.vert");
	Shader fragmentGBufferWrite(ShaderType::FRAGMENT, "Shaders/GWriteGeneric.frag");

	// Scenes
	Scene crySponza(Scene::sponzaFileName);
	Scene cornellBox(Scene::cornellboxFileName);
	scenes.push_back(&crySponza);
	scenes.push_back(&cornellBox);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	// All Init
	// Render Loop
	while(!mainWindow.WindowClosed())
	{
		// Constantly Check Input Scheme Change
		mainWindow.ChangeInputScheme(*inputSchemes[currentInputScheme % inputSchemes.size()]);

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

		// DrawCall
		scenes[currentScene % scenes.size()]->Draw();
		
		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();
	}
	return 0;
}