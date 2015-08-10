/**

CUDA timer class that uses cuda runtime

*/

#ifndef __CUDAVECTOR_H__
#define __CUDAVECTOR_H__

//#include <cuda_runtime.h>
#include <vector>

template<class T>
class CudaVector
{
	private:
		T*				d_data;
		size_t			size;

	protected:
	public:
		// Constructors & Destructor
						CudaVector();
						CudaVector(size_t count);
						CudaVector(const CudaVector&) = delete;
						CudaVector(CudaVector&&)
						~CudaVector();

		// Assignment Operators
		CudaVector&		operator=(const CudaVector&) = delete;
		CudaVector&		operator=(CudaVector&&);
		CudaVector&		operator=(const std::vector<T>&);

		// Generic Copy
		void			Copy(const T* hostPtrStart, size_t count, size_t stride);
};
#endif //__CUDAVECTOR_H__