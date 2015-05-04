#include "Window.h"
#include "Macros.h"
#include "GLHeader.h"
#include "InputManI.h"
#include "Globals.h"

#include <GLFW/glfw3.h>

std::map<GLFWwindow*, Window*> Window::windowMappings;

void Window::ErrorCallbackGLFW(int error, const char* description)
{
	GI_ERROR_LOG("GLFW Error %d: %s", error, description);
}

void Window::WindowPosGLFW(GLFWwindow* w, int width, int  height)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowPosChangedFunc(width, height);
	}
}

void Window::WindowFBGLFW(GLFWwindow* w, int width, int height)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowFBChangedFunc(width, height);
	}
}

void Window::WindowSizeGLFW(GLFWwindow* w, int width, int height)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowSizeChangedFunc(width, height);
	}
}

void Window::WindowCloseGLFW(GLFWwindow* w)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowClosedFunc();
	}
}

void Window::WindowRefreshGLFW(GLFWwindow* w)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowRefreshedFunc();
	}
}

void Window::WindowFocusedGLFW(GLFWwindow* w, int b)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowFocusedFunc((b == 1) ? true : false);
	}
}

void Window::WindowMinimizedGLFW(GLFWwindow* w, int b)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->WindowMinimizedFunc((b == 1) ? true : false);
	}
}

void Window::KeyboardUsedGLFW(GLFWwindow* w, int key, int scancode, int action, int mods)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->KeyboardUsedFunc(key, scancode, action, mods);
	}
}

void Window::MouseMovedGLFW(GLFWwindow* w, double x, double y)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->MouseMovedFunc(x, y);
	}
}

void Window::MousePressedGLFW(GLFWwindow* w, int button, int action, int mods)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->MousePressedFunc(button, action, mods);
	}
}

void Window::MouseScrolledGLFW(GLFWwindow* w, double xoffset, double yoffset)
{
	std::map<GLFWwindow*, Window*>::iterator i;
	i = windowMappings.find(w);
	if(i != windowMappings.end())
	{
		assert(i->second->input != nullptr);
		i->second->input->MouseScrolledFunc(xoffset, yoffset);
	}
}

void __stdcall Window::OGLCallbackRender(GLenum, 
										 GLenum type,
										 GLuint id,
										 GLenum severity,
										 GLsizei,
										 const GLchar* message,
										 const void*)
{
	// Dont Show Others For Now
	if(type == GL_DEBUG_TYPE_OTHER)
		return;

	GI_DEBUG_LOG("---------------------OGL-Callback-Render------------");
	GI_DEBUG_LOG("Message: %s", message);
	switch(type)
	{
		case GL_DEBUG_TYPE_ERROR:
			GI_DEBUG_LOG("Type: ERROR");
			break;
		case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR:
			GI_DEBUG_LOG("Type: DEPRECATED_BEHAVIOR");
			break;
		case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:
			GI_DEBUG_LOG("Type: UNDEFINED_BEHAVIOR");
			break;
		case GL_DEBUG_TYPE_PORTABILITY:
			GI_DEBUG_LOG("Type: PORTABILITY");
			break;
		case GL_DEBUG_TYPE_PERFORMANCE:
			GI_DEBUG_LOG("Type: PERFORMANCE");
			break;
		case GL_DEBUG_TYPE_OTHER:
			GI_DEBUG_LOG("Type: OTHER");
			break;
	}

	GI_DEBUG_LOG("ID: %d", id);
	switch(severity)
	{
		case GL_DEBUG_SEVERITY_LOW:
			GI_DEBUG_LOG("Severity: LOW");
			break;
		case GL_DEBUG_SEVERITY_MEDIUM:
			GI_DEBUG_LOG("Severity: MEDIUM");
			break;
		case GL_DEBUG_SEVERITY_HIGH:
			GI_DEBUG_LOG("Severity: HIGH");
			break;
		default:
			GI_DEBUG_LOG("Severity: NONE");
			break;
	}
	GI_DEBUG_LOG("---------------------OGL-Callback-Render-End--------------");
}


Window::Window(InputManI& input,
			   WindowProperties properties)
	: input(&input)	
	, window(nullptr)
{
	// If you are first window init glfw
	if(windowMappings.empty())
	{
		if(!glfwInit())
		{
			GI_ERROR_LOG("Fatal Error: Could not Init GLFW");
			assert(false);
		}
		glfwSetErrorCallback(ErrorCallbackGLFW);
	}

	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

	glfwWindowHint(GLFW_RED_BITS, 8);
	glfwWindowHint(GLFW_GREEN_BITS, 8);
	glfwWindowHint(GLFW_BLUE_BITS, 8);
	glfwWindowHint(GLFW_ALPHA_BITS, 8);

	glfwWindowHint(GLFW_DEPTH_BITS, 24);
	glfwWindowHint(GLFW_STENCIL_BITS, 8);

	glfwWindowHint(GLFW_SAMPLES, 8);

	glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_API);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);

	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	glfwWindowHint(GLFW_CONTEXT_RELEASE_BEHAVIOR, GLFW_RELEASE_BEHAVIOR_NONE);

	if(DEBUG)
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);
	else
		glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_FALSE);

	switch(properties.screenType)
	{
		case WindowScreenType::BORDERLESS:
		{
			const GLFWvidmode* mode = glfwGetVideoMode(glfwGetPrimaryMonitor());
			glfwWindowHint(GLFW_RED_BITS, mode->redBits);
			glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
			glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
			window = glfwCreateWindow(mode->width, mode->height, "GI Thesis", glfwGetPrimaryMonitor(), nullptr);
			break;
		}
		case WindowScreenType::WINDOWED:
		{
			window = glfwCreateWindow(properties.width, properties.height, "GI Thesis", nullptr, nullptr);
			break;
		}
		case WindowScreenType::FULLSCREEN:
		{
			window = glfwCreateWindow(properties.width, properties.height, "GI Thesis", glfwGetPrimaryMonitor(), nullptr);
			break;
		}
		default:
			break;
	}

	if(window == nullptr)
	{
		GI_ERROR_LOG("Fatal Error: Could not create window.");
		assert(false);
	}

	glfwMakeContextCurrent(window);

	// Now Init GLEW
	glewExperimental = GL_TRUE;
	GLenum err = glewInit();
	if(err != GLEW_OK)
	{
		GI_ERROR_LOG("Error: %s\n", glewGetErrorString(err));
		assert(false);
	}

	// Print Stuff Now
	// Window Done
	GI_LOG("Window Initialized.");
	GI_LOG("GLEW\t: %s", glewGetString(GLEW_VERSION));
	GI_LOG("GLFW\t: %s", glfwGetVersionString());
	GI_LOG("");
	GI_LOG("Renderer Information...");
	GI_LOG("OpenGL\t: %s", glGetString(GL_VERSION));
	GI_LOG("GLSL\t: %s", glGetString(GL_SHADING_LANGUAGE_VERSION));
	GI_LOG("Device\t: %s", glGetString(GL_RENDERER));
	GI_LOG("");

	if(DEBUG)
	{
		// Add Callback
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		glDebugMessageCallback(Window::OGLCallbackRender, nullptr);
		glDebugMessageControl(GL_DONT_CARE,
							  GL_DONT_CARE,
							  GL_DONT_CARE,
							  0,
							  nullptr,
							  GL_TRUE);
	}

	// Get Some GPU Limitations
	// DEBUG
	GLint uniformBufferOffsetAlignment, ssbOffsetAlignment;
	glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &uniformBufferOffsetAlignment);
	glGetIntegerv(GL_SHADER_STORAGE_BUFFER_OFFSET_ALIGNMENT, &ssbOffsetAlignment);

	// Set Callbacks
	glfwSetWindowPosCallback(window, Window::WindowPosGLFW);
	glfwSetFramebufferSizeCallback(window, Window::WindowFBGLFW);
	glfwSetWindowSizeCallback(window, Window::WindowSizeGLFW);
	glfwSetWindowCloseCallback(window, Window::WindowCloseGLFW);
	glfwSetWindowRefreshCallback(window, Window::WindowRefreshGLFW);
	glfwSetWindowFocusCallback(window, Window::WindowFocusedGLFW);
	glfwSetWindowIconifyCallback(window, Window::WindowMinimizedGLFW);

	glfwSetKeyCallback(window, Window::KeyboardUsedGLFW);
	glfwSetCursorPosCallback(window, Window::MouseMovedGLFW);
	glfwSetMouseButtonCallback(window, Window::MousePressedGLFW);
	glfwSetScrollCallback(window, Window::MouseScrolledGLFW);

	glfwSwapInterval(0);

	windowMappings.insert(std::make_pair(window, this));
	glfwShowWindow(window);
}

Window::~Window()
{
	glfwMakeContextCurrent(nullptr);
	glfwDestroyWindow(window);

	// If you are last window alive also close up glfw
	windowMappings.erase(window);
	if(windowMappings.empty())
	{
		glfwTerminate();
	}
}

void Window::ChangeInputScheme(InputManI& newInput)
{
	// Adress check is fine here
	if(&newInput != input)
		input = &newInput;
}

bool Window::WindowClosed() const
{
	return glfwWindowShouldClose(window) == 0 ? false : true;
}

void Window::Present()
{
	glfwSwapBuffers(window);
}