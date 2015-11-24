#include "CudaInit.h"
#include "Macros.h"
#include "GIKernels.cuh"

cudaDeviceProp CudaInit::props = {};
bool CudaInit::init = false;

void CudaInit::InitCuda()
{
	// Setting Device
	cudaSetDevice(0);

	// Cuda Check
	CUDA_CHECK(cudaGetDeviceProperties(&props, 0));

	// Info Print
	GI_LOG("Cuda Information...");
	GI_LOG("GPU Name\t\t: %s", props.name);
	GI_LOG("GPU Compute Capability\t: %d%d", props.major, props.minor);
	GI_LOG("GPU SM Count: %d", props.multiProcessorCount);
	GI_LOG("GPU Shared Memory(SM)\t: %dKB", props.sharedMemPerMultiprocessor / 1024);
	GI_LOG("GPU Shared Memory(Block): %dKB", props.sharedMemPerBlock / 1024);
	GI_LOG("");

	// Minimum Required Compute Capability
	if(props.major < 3)
	{
		GI_LOG("#######################################################################");
		GI_LOG("UNSUPPORTED GPU, CUDA PORTION WILL NOT WORK. NEEDS ATLEAST SM_30 DEVICE");
		GI_LOG("#######################################################################");
		GI_LOG("");
	}

	// Shared Memory Prep
	// 16 Kb memory is enough for our needs most of the time
	// or 8kb (for %100 occupancy)
	CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// Kernel Specifics
	CUDA_CHECK(cudaFuncSetCacheConfig(VoxelTransform, cudaFuncCachePreferEqual));

	// All done!
	init = true;
}

unsigned int CudaInit::CapabilityMajor()
{
	assert(init);
	return props.major;
}

unsigned int CudaInit::CapabilityMinor()
{
	assert(init);
	return props.minor;
}

unsigned int CudaInit::SMCount()
{
	assert(init);
	return static_cast<unsigned int>(props.multiProcessorCount);
}