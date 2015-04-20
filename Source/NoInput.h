/**

No Input

Disables Keyboard and Mouse


*/


#ifndef __NOINPUT_H__
#define __NOINPUT_H__

#include "WindowInput.h"

class NoInput : public WindowInput
{
	private:
	protected:
	public:
						NoInput(Camera& cam) : WindowInput(cam) {}

		virtual void	KeyboardUsedFunc(int, int, int, int) override {};
		virtual void	MouseMovedFunc(double, double) override {};
		virtual void	MousePressedFunc(int, int, int) override {};
		virtual void	MouseScrolledFunc(double, double) override {};

};
#endif //__NOINPUT_H__