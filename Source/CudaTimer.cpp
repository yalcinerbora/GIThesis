#include <cassert>
#include <cstdio>
#include "CudaTimer.h"
#include "CudaInit.h"

CudaTimer::CudaTimer(cudaStream_t stream)
	: start(nullptr)
	, end(nullptr)
	, stream(stream)
{
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&end));
}

CudaTimer::~CudaTimer()
{
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(end));
}

void CudaTimer::Start()
{
	CUDA_CHECK(cudaEventRecord(start, stream));
}

void CudaTimer::Stop()
{
	CUDA_CHECK(cudaEventRecord(end, stream));
}

double CudaTimer::ElapsedS()
{
	float ms = 0;
	CUDA_CHECK(cudaEventSynchronize(end));
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
	return static_cast<double>(ms) * 0.001;
}

double CudaTimer::ElapsedMilliS()
{
	float ms = 0;
	CUDA_CHECK(cudaEventSynchronize(end));
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
	return static_cast<double>(ms);
}

double CudaTimer::ElapsedMicroS()
{
	float ms = 0;
	CUDA_CHECK(cudaEventSynchronize(end));
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
	return static_cast<double>(ms) * 1000.0;
}

double CudaTimer::ElapsedNanoS()
{
	float ms = 0;
	CUDA_CHECK(cudaEventSynchronize(end));
	CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
	return static_cast<double>(ms) * 1000.0 * 1000.0;
}