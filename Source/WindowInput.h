/**

Base Class For Input Management

Handles Close Minimize callbacks

Does not interfere with keyboard mouse input

*/


#ifndef __WINDOWINPUT_H__
#define __WINDOWINPUT_H__

#include "InputManI.h"
#include "ArrayStruct.h"
#include  <vector>

struct Camera;
class SolutionI;
class SceneI;
class Window;

class WindowInput : public InputManI
{
	private:
		Camera&				camera;
		uint32_t&			currentSolution;
		uint32_t&			currentScene;
		uint32_t&			currentInput;

	protected:
	public:
							WindowInput(Camera& cam,
										uint32_t& currentSolution,
										uint32_t& currentScene,
										uint32_t& currentInput);

		virtual void		WindowPosChangedFunc(int posX, int posY) override;
		virtual void		WindowFBChangedFunc(int fbWidth, int fbHeight) override;
		virtual void		WindowSizeChangedFunc(int width, int height) override;
		virtual void		WindowClosedFunc() override;
		virtual void		WindowRefreshedFunc() override;
		virtual void		WindowFocusedFunc(bool) override;
		virtual void		WindowMinimizedFunc(bool) override;

		// Explicitly showing un-implemented functions
		virtual void		KeyboardUsedFunc(int key, int osKey, int action, int modifier);
		virtual void		MouseMovedFunc(double x, double y);
		virtual void		MousePressedFunc(int button, int action, int modifier);
		virtual void		MouseScrolledFunc(double xOffset, double yOffset);
};

#endif //__WINDOWINPUT_H__