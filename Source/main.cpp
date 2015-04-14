#include <iostream>
#include <GFG/GFGHeader.h>

#include "Window.h"
#include "GLFW/glfw3.h"

#include "NoInput.h"
#include "FPSInput.h"
#include "MayaInput.h"

#include "Globals.h"

int main()
{
	// Input Schemes
	NoInput nullInput;		// No Input from peripheral devices
	FPSInput fpsInput;
	MayaInput mayaInput;

	// Window Init
	WindowProperties winProps
	{
		1280,
		720,
		WindowScreenType::WINDOWED
	};
	Window mainWindow(nullInput,
					  winProps);













	// All Init
	// Render Loop
	while(!mainWindow.WindowClosed())
	{
		// Here Start A Implementation
		// Implementation is a 


		




		// End of the Loop
		mainWindow.Present();
		glfwPollEvents();
	}

	return 0;
}