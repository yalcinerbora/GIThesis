/**

InputManI

InputManagement Interface for decoupling window and input

*/

#ifndef __INPUTMANI_H__
#define __INPUTMANI_H__

// Window Input
//typedef std::function<void(int, int)> WindowPosChangedFunc;
//typedef std::function<void(int, int)> WindowFBChangedFunc;
//typedef std::function<void(int, int)> WindowSizeChangedFunc;
//typedef std::function<void(void)> WindowClosedFunc;
//typedef std::function<void(void)> WindowRefreshedFunc;
//typedef std::function<void(bool)> WindowFocusedFunc;
//typedef std::function<void(bool)> WindowMinimizedFunc;
//
//// Peripheral Device Input
//typedef std::function<void(int, int, int, int)> KeyboardUsedFunc;
//typedef std::function<void(double, double)> MouseMovedFunc;
//typedef std::function<void(int, int, int)> MousePressedFunc;
//typedef std::function<void(double, double)> MouseScrolledFunc;

// TODO: Add Joystick later
class InputManI
{
	private:
	protected:
	public:
		virtual			~InputManI() = default;

		virtual void	WindowPosChangedFunc(int, int) = 0;
		virtual void	WindowFBChangedFunc(int, int) = 0;
		virtual void	WindowSizeChangedFunc(int, int) = 0;
		virtual void	WindowClosedFunc() = 0;
		virtual void	WindowRefreshedFunc() = 0;
		virtual void	WindowFocusedFunc(bool) = 0;
		virtual void	WindowMinimizedFunc(bool) = 0;
        virtual bool    MoveLight() const = 0;
        virtual bool    Movement() const = 0;

		virtual void	KeyboardUsedFunc(int, int, int, int) = 0;
		virtual void	MouseMovedFunc(double, double) = 0;
		virtual void	MousePressedFunc(int, int, int) = 0;
		virtual void	MouseScrolledFunc(double, double) = 0;

		virtual void	AddKeyCallback(int, int, void(*)(void*), void*) = 0;
};

#endif //__INPUTMANI_H__