#pragma once
/**

Timer Class. wrapping std::chrono
Simple start, stop then check difference

Author(s):
	Bora Yalciner

*/

#include <chrono>

// JESUS THAT NAMESPACES+TEMPLATES
using IETimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;
using IEDuration = std::chrono::duration<double>;

class IETimer 
{
	private:
		IETimePoint		start;
		IEDuration		elapsed;

	protected:

	public:
		//Constructors & Destructor
						IETimer();
						IETimer(const IETimer&) = default;
						~IETimer() = default;

		// Functionality
		void			Start();
		void			Stop();
		void			Lap();

		// Elapsed Time Between Start And Stop
		double			ElapsedS();
		double			ElapsedMilliS();
		double			ElapsedMicroS();
		double			ElapsedNanoS();
};


