/**

Base Class For Input Management

Handles Close Minimize callbacks

Does not interfere with keyboard mouse input

*/


#ifndef __WINDOWINPUT_H__
#define __WINDOWINPUT_H__

#include "InputManI.h"
#include "ArrayStruct.h"
#include "Camera.h"
#include <map>

class SolutionI;
class SceneI;
class Window;

using CallbackArray = std::multimap<std::pair<int, int>, 
									std::pair<void(*)(void*), void*>>;



class WindowInput : public InputManI
{
	private:
		uint32_t&				currentSolution;
		uint32_t&				currentScene;
		uint32_t&				currentInput;
		
		CallbackArray			callbacks;
        bool                    moveLight;
        bool                    movement;

	protected:
		Camera&					camera;
		static const Camera		savedCamera;

	public:
							WindowInput(Camera& cam,
										uint32_t& currentSolution,
										uint32_t& currentScene,
										uint32_t& currentInput);

		void				WindowPosChangedFunc(int posX, int posY) override;
		void				WindowFBChangedFunc(int fbWidth, int fbHeight) override;
		void				WindowSizeChangedFunc(int width, int height) override;
		void				WindowClosedFunc() override;
		void				WindowRefreshedFunc() override;
		void				WindowFocusedFunc(bool) override;
		void				WindowMinimizedFunc(bool) override;
		void				AddKeyCallback(int, int, void(*)(void*), void*) override;
        bool                MoveLight() const override;
        bool                Movement() const override;

		virtual void		KeyboardUsedFunc(int key, int osKey, int action, int modifier);
		virtual void		MouseMovedFunc(double x, double y);
		virtual void		MousePressedFunc(int button, int action, int modifier);
		virtual void		MouseScrolledFunc(double xOffset, double yOffset);
};

#endif //__WINDOWINPUT_H__