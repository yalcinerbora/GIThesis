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
			CUDA_CHECK(cudaDeviceSynchronize()); \
			CUDA_CHECK(cudaPeekAtLastError());
			
#else
	#define CUDA_CHECK(func) func;
	#define CUDA_KERNEL_CHECK()
#endif

class CudaInit
{
	public:
		// Thread per Block counts
		static constexpr int	TBPSmall = 128;
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

};
#endif //__CUDAINIT_H__
