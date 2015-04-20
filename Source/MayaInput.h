/**

Maya Like Camera Movement

uses LeftMouse button instead of ALT key.
other than that it is comparable to Autodesk Maya Input

*/


#ifndef __MAYAINPUT_H__
#define __MAYAINPUT_H__

#include "WindowInput.h"

class MayaInput : public WindowInput
{
	private:
	protected:
	public:
						MayaInput(Camera& cam) : WindowInput(cam) {}

		virtual void	KeyboardUsedFunc(int, int, int, int) override;
		virtual void	MouseMovedFunc(double, double) override;
		virtual void	MousePressedFunc(int, int, int) override;
		virtual void	MouseScrolledFunc(double, double) override;

};
#endif //__MAYAINPUT_H__