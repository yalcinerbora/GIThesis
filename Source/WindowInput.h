#pragma once
/**

Base Class For Input Management

Handles Close Minimize callbacks, Scene and Solution Changes

Does not interfere with camera movement input delegates it to selected Camera Interface

*/

#include <functional>
#include <vector>
#include <map>

#include "CameraInputI.h"
#include "Camera.h"

class SolutionI;
class SceneI;
class Window;

using CallbackArray = std::multimap<std::pair<int, int>, 
									std::function<void()>>;

class WindowInput
{
	private:
		Camera&								camera;
		CallbackArray						callbacks;
		const std::vector<CameraInputI*>&	cameraInputs;
		const std::vector<SceneI*>			scenes;
		const std::vector<SolutionI*>		solutions;
		uint32_t							currentCameraInput;
		uint32_t							currentSolution;
		uint32_t							currentScene;

	protected:
	public:
											WindowInput(Camera&,
														const std::vector<CameraInputI*>& cameraInputs,
														const std::vector<SolutionI*>& solutions,
														const std::vector<SceneI*>& scenes);

		void								WindowPosChangedFunc(int posX, int posY);
		void								WindowFBChangedFunc(int fbWidth, int fbHeight);
		void								WindowSizeChangedFunc(int width, int height);
		void								WindowClosedFunc();
		void								WindowRefreshedFunc();
		void								WindowFocusedFunc(bool);
		void								WindowMinimizedFunc(bool);

		//
		virtual void						KeyboardUsedFunc(int key, int osKey, int action, int modifier);
		virtual void						MouseMovedFunc(double x, double y);
		virtual void						MousePressedFunc(int button, int action, int modifier);
		virtual void						MouseScrolledFunc(double xOffset, double yOffset);
		
		// Fetch
		SolutionI*							Solution();
		SceneI*								Scene();

		// Defining Custom Callback
		template <class Function, class... Args>
		void								AddKeyCallback(int, int, Function&& f, Args&&... args);
};

template <class Function, class... Args>
void WindowInput::AddKeyCallback(int glfwKey, int glfwAction, Function&& f, Args&&... args)
{
	std::function<void()> func = std::bind(f, args...);
	callbacks.emplace(glfwKey, glfwAction, func);
}