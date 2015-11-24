/**

Cuda Init Class

Determines shared memory modes etc.


*/

#ifndef __CUDAINIT_H__
#define __CUDAINIT_H__

#include <cuda_runtime.h>

#ifdef GI_DEBUG
	#define CUDA_CHECK(func) \
			{ \
				cudaError_t err;  \
				if((err = func) != cudaSuccess) \
				{ \
					printf("Error: \"%s\"\n", cudaGetErrorString(err)); \
					assert(false); \
				} \
			}
#define CUDA_KERNEL_CHECK() \
			CUDA_CHECK(cudaPeekAtLastError()); \
			CUDA_CHECK(cudaDeviceSynchronize());
#else
	#define CUDA_CHECK(func) func;
	#define CUDA_KERNEL_CHECK()
#endif

class CudaInit
{
	private:
		static cudaDeviceProp	props;
		static bool				init;

	public:
		static void				InitCuda();
		static unsigned int		CapabilityMajor();
		static unsigned int		CapabilityMinor();
		static unsigned int		SMCount();

};
#endif //__CUDAINIT_H__
