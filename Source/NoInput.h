/**

No Input

Disables Keyboard and Mouse


*/


#ifndef __NOINPUT_H__
#define __NOINPUT_H__

#include "WindowInput.h"

class NoInput : public CameraInputI
{
	private:
		static const std::string	NoInputName;

	protected:
	public:
		 void						KeyboardUsedFunc(Camera&, int, int, int, int) override {}
		 void						MouseMovedFunc(Camera&, double, double) override {}
		 void						MousePressedFunc(Camera&, int, int, int) override {}
		 void						MouseScrolledFunc(Camera&, double, double) override {}

		 const std::string&			Name() const override;
};

const std::string NoInput::NoInputName = "NoInput";

const inline std::string& NoInput::Name() const
{
	return NoInputName;
}
#endif //__NOINPUT_H__