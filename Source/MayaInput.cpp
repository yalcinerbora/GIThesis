#include "MayaInput.h"

MayaInput::MayaInput(Camera& cam,
					uint32_t& currentSolution,
					uint32_t& currentScene,
					uint32_t& currentInput)
	: WindowInput(cam, 
				  currentSolution, 
				  currentScene, 
				  currentInput)
{
	
}

void MayaInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{
	WindowInput::KeyboardUsedFunc(key, osKey, action, modifier);
}

void MayaInput::MouseMovedFunc(double x, double y)
{
	WindowInput::MouseMovedFunc(x, y);
}

void MayaInput::MousePressedFunc(int button, int action, int modifier)
{
	WindowInput::MousePressedFunc(button, action, modifier);
}

void MayaInput::MouseScrolledFunc(double xOffset, double yOffset)
{
	WindowInput::MouseScrolledFunc(xOffset, yOffset);
}