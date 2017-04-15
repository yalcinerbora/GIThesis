/**

Maya Like Camera Movement

uses LeftMouse button instead of ALT key.
other than that it is comparable to Autodesk Maya Input

*/


#ifndef __MAYAINPUT_H__
#define __MAYAINPUT_H__

#include "CameraInputI.h"

class MayaInput : public CameraInputI
{
	private:
		static const std::string	MayaInputName;

		const double				sensitivity;
		const double				zoomPercentage;
		const double				translateModifier;

		bool						moveMode;
		bool						translateMode;
		double						mouseX;
		double						mouseY;

	protected:
	public:
									MayaInput(double sensitivity, 
											  double zoomPercentage, 
											  double translateModifier);

		virtual void				KeyboardUsedFunc(Camera&, int, int, int, int) override;
		virtual void				MouseMovedFunc(Camera&, double, double) override;
		virtual void				MousePressedFunc(Camera&, int, int, int) override;
		virtual void				MouseScrolledFunc(Camera&, double, double) override;

		const std::string&			Name() const override;
};
#endif //__MAYAINPUT_H__