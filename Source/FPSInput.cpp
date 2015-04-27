#include "FPSInput.h"

FPSInput::FPSInput(Camera& cam,
					 uint32_t& currentSolution,
					 uint32_t& currentScene,
					 uint32_t& currentInput)
	: WindowInput(cam,
				  currentSolution,
				  currentScene,
				  currentInput)
{}

void FPSInput::KeyboardUsedFunc(int key, int osKey, int action, int modifier)
{
	WindowInput::KeyboardUsedFunc(key, osKey, action, modifier);
}

void FPSInput::MouseMovedFunc(double x, double y)
{
	WindowInput::MouseMovedFunc(x, y);
}

void FPSInput::MousePressedFunc(int button, int action, int modifier)
{
	WindowInput::MousePressedFunc(button, action, modifier);
}

void FPSInput::MouseScrolledFunc(double xOffset, double yOffset)
{
	WindowInput::MouseScrolledFunc(xOffset, yOffset);
}