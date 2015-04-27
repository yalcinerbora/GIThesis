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
						NoInput(Camera& cam,
								uint32_t& currentSolution,
								uint32_t& currentScene,
								uint32_t& currentInput) : WindowInput(cam,
																	  currentSolution,
																	  currentScene,
																	  currentInput) {}
};
#endif //__NOINPUT_H__