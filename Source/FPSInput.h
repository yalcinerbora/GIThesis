/**

FPS Gamelike Camera Movement

Wasd to move, mouse to look,
LMB locks and hids the mouse to the center

*/


#ifndef __FPSINPUT_H__
#define __FPSINPUT_H__

#include "WindowInput.h"

class FPSInput : public CameraInputI
{
	private:
		static const std::string    FPSInputName;

		const double				sensitivity;
		const double				moveRatio;
		const double				moveRatioModifier;
		
		bool						fpsMode;
		double						mouseX;
		double						mouseY;
		double						moveRatioModified;

	protected:
	public:
									FPSInput(double sensitivity, 
											 double moveratio, 
											 double moveRatioModifier);

		virtual void				KeyboardUsedFunc(Camera&, int, int, int, int) override;
		virtual void				MouseMovedFunc(Camera&, double, double) override;
		virtual void				MousePressedFunc(Camera&, int, int, int) override;
		virtual void				MouseScrolledFunc(Camera&, double, double) override;

		const std::string&			Name() const override;
};
#endif //__FPSINPUT_H__