/**

Window Management Using GLFW

Requires Input Management Interface to get the incoming Data


*/

#ifndef __WINDOW_H__
#define __WINDOW_H__

#include <map>
#include "GLHeaderLite.h"

class InputManI;
struct GLFWwindow;

enum class WindowScreenType
{
	FULLSCREEN,
	WINDOWED,
	BORDERLESS
};

struct WindowProperties
{
	int					width;
	int					height;
	WindowScreenType	screenType;
	
};

class Window
{
	private:
		// Static Properties
		static std::map<GLFWwindow*, Window*> windowMappings;

		// Class Properties	
		InputManI*				input;
		GLFWwindow*				window;

		// GLFWCallbacks
		static void				ErrorCallbackGLFW(int, const char*);
		static void				WindowPosGLFW(GLFWwindow*, int, int);
		static void				WindowFBGLFW(GLFWwindow*, int, int);
		static void				WindowSizeGLFW(GLFWwindow*, int, int);
		static void				WindowCloseGLFW(GLFWwindow*);
		static void				WindowRefreshGLFW(GLFWwindow*);
		static void				WindowFocusedGLFW(GLFWwindow*, int);
		static void				WindowMinimizedGLFW(GLFWwindow*, int);

		static void				KeyboardUsedGLFW(GLFWwindow*, int, int, int, int);
		static void				MouseMovedGLFW(GLFWwindow*, double, double);
		static void				MousePressedGLFW(GLFWwindow*, int, int, int);
		static void				MouseScrolledGLFW(GLFWwindow*, double, double);

		// OGL Debug Context Callback
		static void _stdcall	OGLCallbackRender(GLenum source, GLenum type,
												  GLuint id,
												  GLenum severity,
												  GLsizei length,
												  const GLchar* message,
												  void* userParam);

	protected:
	public:
		// Constructors & Destructor
								Window(InputManI&,
									   WindowProperties);
								Window(const Window&) = delete;
		const Window&			operator=(const Window&) = delete;
								~Window();

		// Utility
		// Change Input Scheme
		void					ChangeInputScheme(InputManI&);
		bool					WindowClosed() const;
		void					Present();

};
#endif //__WINDOW_H__