#include "IETimer.h"

IETimer::IETimer() : start(std::chrono::high_resolution_clock::now()),
						elapsed(std::chrono::seconds(0))
{}

void IETimer::Start()
{
	start = std::chrono::high_resolution_clock::now();
}

void IETimer::Stop()
{
	elapsed = std::chrono::high_resolution_clock::now() - start;
}

void IETimer::Lap()
{
	Stop();
	Start();
}

// Elapsed Time Between Start And Stop
double IETimer::ElapsedS()
{
	return elapsed.count();
}

double IETimer::ElapsedMilliS()
{
	return elapsed.count() * 1000;
}

double IETimer::ElapsedMicroS()
{
	return elapsed.count() * 1000 * 1000;
}

double IETimer::ElapsedNanoS()
{
	return elapsed.count() * 1000 * 1000 * 1000;
}