/**

OGL Context Creation


*/

#ifndef __ASCIIPROGRESSBAR_H__
#define __ASCIIPROGRESSBAR_H__

#include <algorithm>
#include "Macros.h"

#define BAR_DEFAULT_WIDTH 64

class ASCIIProgressBar
{
	private:
		int				stepCount;
		unsigned int	currentStep;
		int				width;

	protected:
	public:
						// Constructors & Destructor
						ASCIIProgressBar();
						ASCIIProgressBar(unsigned int steps, int width = BAR_DEFAULT_WIDTH);
		
		void			Reset();
		void			Step();
		void			Print();
		void			End();


	
};
ASCIIProgressBar::ASCIIProgressBar()
	: stepCount(100)
	, width(width)
	, currentStep(0)
{}

ASCIIProgressBar::ASCIIProgressBar(unsigned int steps, int width)
	: stepCount(steps)
	, width(width)
	, currentStep(0)
{}

void ASCIIProgressBar::Reset()
{
	currentStep = 0;
}

void ASCIIProgressBar::Step()
{
	currentStep++;
	currentStep = std::min<unsigned int>(currentStep, stepCount);
}

void ASCIIProgressBar::Print()
{
	float ratio = static_cast<float>(currentStep) / stepCount;
	int percent = static_cast<int>(std::floor(ratio * 100.0f));
	int count = static_cast<int>(std::floor(ratio * width));

	std::cout << "[";
	for(int i = 0; i < count; i++)
		std::cout << "#";
	for(int i = 0; i < width - count; i++)
		std::cout << " ";
	std::cout << "] %%";
	std::cout << "%d\r", percent;
	std::cout.flush();
}

void ASCIIProgressBar::End()
{
	//for(int i = 0; i < width + 10; i++)
	//	printf(" ");
	GI_LOG("");
}
#endif //__ASCIIPROGRESSBAR_H__