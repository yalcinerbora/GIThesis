/**

Cuda Init Class

Determines shared memory modes etc.


*/

#ifndef __CUDAINIT_H__
#define __CUDAINIT_H__

#include <cuda_runtime.h>
#include "Macros.h"

#ifdef GI_DEBUG
	#define CUDA_CHECK(func) {CudaInit::GPUAssert((func), __FILE__, __LINE__);}
	#define CUDA_KERNEL_CHECK() \
		CUDA_CHECK(cudaGetLastError()); \
		CUDA_CHECK(cudaDeviceSynchronize());
#else
	#define CUDA_CHECK(func) func;
	#define CUDA_KERNEL_CHECK()
#endif

class CudaInit
{
	public:
		// Thread per Block counts
		static constexpr int	TBPSmall = 256;
		static constexpr int	TBP = 512;
		static constexpr int	TBP_XY = 16;

		

	private:
		static cudaDeviceProp	props;
		static bool				init;

	public:
		static void				InitCuda();
		static unsigned int		CapabilityMajor();
		static unsigned int		CapabilityMinor();
		static unsigned int		SMCount();

		static int				GenBlockSize(int totalThread);
		static int				GenBlockSizeSmall(int totalThread);
		static int2				GenBlockSize2D(int2 totalThread);

		static void				GPUAssert(cudaError_t code, 
										  const char *file, 
										  int line);

};

inline void CudaInit::GPUAssert(cudaError_t code, const char *file, int line)
{
	if(code != cudaSuccess)
	{
		GI_ERROR_LOG("Cuda Failure: %s %s %d\n", cudaGetErrorString(code), file, line);
		assert(false);
	}
}
#endif //__CUDAINIT_H__
