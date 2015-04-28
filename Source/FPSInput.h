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
		static double	Sensitivity;
		static double	MoveRatio;
		
		bool			FPSMode;
		double			mouseX;
		double			mouseY;

	protected:
	public:
						FPSInput(Camera& cam,
										uint32_t& currentSolution,
										uint32_t& currentScene,
										uint32_t& currentInput);

		virtual void	KeyboardUsedFunc(int, int, int, int) override;
		virtual void	MouseMovedFunc(double, double) override;
		virtual void	MousePressedFunc(int, int, int) override;
		virtual void	MouseScrolledFunc(double, double) override;

};
#endif //__FPSINPUT_H__