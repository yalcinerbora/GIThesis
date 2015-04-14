/**

FPS Gamelike Camera Movement

Wasd to move, mouse to look,
LMB locks and hids the mouse to the center

*/


#ifndef __FPSINPUT_H__
#define __FPSINPUT_H__

#include "WindowInput.h"

class FPSInput : public WindowInput
{
	private:
	protected:
	public:

		virtual void	KeyboardUsedFunc(int, int, int, int) override;
		virtual void	MouseMovedFunc(double, double) override;
		virtual void	MousePressedFunc(int, int, int) override;
		virtual void	MouseScrolledFunc(double, double) override;

};
#endif //__FPSINPUT_H__