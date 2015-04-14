/**

Base Class For Input Management

Handles Close Minimize callbacks

Does not interfere with keyboard mouse input

*/


#ifndef __WINDOWINPUT_H__
#define __WINDOWINPUT_H__

#include "InputManI.h"

class WindowInput : public InputManI
{
	private:
	protected:
	public:
		virtual void	WindowPosChangedFunc(int, int) override;
		virtual void	WindowFBChangedFunc(int, int) override;
		virtual void	WindowSizeChangedFunc(int, int) override;
		virtual void	WindowClosedFunc() override;
		virtual void	WindowRefreshedFunc() override;
		virtual void	WindowFocusedFunc(bool) override;
		virtual void	WindowMinimizedFunc(bool) override;

		// Explicitly showing un-implemented functions
		virtual void	KeyboardUsedFunc(int, int, int, int) = 0;
		virtual void	MouseMovedFunc(double, double) = 0;
		virtual void	MousePressedFunc(int, int, int) = 0;
		virtual void	MouseScrolledFunc(double, double) = 0;
		
};

#endif //__WINDOWINPUT_H__