/**

Cuda Definitions

*/


#ifndef __CUDADEFINITIONS_H__
#define __CUDADEFINITIONS_H__

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
#else
	#define CUDA_CHECK(func) func;
#endif


#endif