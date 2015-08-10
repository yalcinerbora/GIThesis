/**

CUDA timer class that uses cuda runtime

*/

#ifndef __CUDATIMER_H__
#define __CUDATIMER_H__

#include <cuda_runtime.h>

class CudaTimer
{
	private:
		cudaEvent_t			start;
		cudaEvent_t			end;
		cudaStream_t		stream;

	protected:

	public:
		// Constructors & Destructor
						CudaTimer(cudaStream_t stream = (cudaStream_t)0);
						CudaTimer(const CudaTimer&) = delete;
		CudaTimer&		operator=(const CudaTimer&) = delete;
						~CudaTimer();

		// Functionality
		void			Start();
		void			Stop();

		// Elapsed Time Between Start And Stop
		double			ElapsedS();
		double			ElapsedMilliS();
		double			ElapsedMicroS();
		double			ElapsedNanoS();
};
#endif //__CUDATIMER_H__