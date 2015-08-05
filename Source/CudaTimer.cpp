#include "CudaTimer.h"

CudaTimer::CudaTimer(cudaStream_t stream)
	: start(nullptr)
	, end(nullptr)
	, stream(stream)
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
}

CudaTimer::~CudaTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}

void CudaTimer::Start()
{
	cudaEventRecord(start, stream);
}

void CudaTimer::Stop()
{
	cudaEventRecord(end, stream);
}

void CudaTimer::Lap()
{
	Stop();
	Start();
}

double CudaTimer::ElapsedS()
{
	float ms = 0;
	cudaEventElapsedTime(&ms, start, end);
	return static_cast<double>(ms) * 0.001;
}

double CudaTimer::ElapsedMilliS()
{
	float ms = 0;
	cudaEventElapsedTime(&ms, start, end);
	return static_cast<double>(ms);
}

double CudaTimer::ElapsedMicroS()
{
	float ms = 0;
	cudaEventElapsedTime(&ms, start, end);
	return static_cast<double>(ms) * 1000.0;
}

double CudaTimer::ElapsedNanoS()
{
	float ms = 0;
	cudaEventElapsedTime(&ms, start, end);
	return static_cast<double>(ms) * 1000.0 * 1000.0;
}