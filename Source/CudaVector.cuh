/**

CUDA timer class that uses cuda runtime

*/

#ifndef __CUDAVECTOR_H__
#define __CUDAVECTOR_H__

#include <cuda_runtime.h>
#include <vector>
#include <type_traits>

#define CUDA_VEC_INITIAL_CAPACITY 256
#define CUDA_VEC_RESIZE_FACTOR 2.0f

template<class T>
class CudaVector
{
	static_assert(std::is_pod<T>::value, "T must be POD for CudaVector");

	private:
		T*						d_data;
		size_t					size;
		size_t					capacity;

		void					ExtendStorage();

	protected:
	public:
		// Constructors & Destructor
						CudaVector();
						CudaVector(size_t count);
						CudaVector(const CudaVector<T>&);
						CudaVector(CudaVector<T>&&);
						~CudaVector();

		// Assignment Operators
		CudaVector<T>&	operator=(const CudaVector<T>&) = delete;
		CudaVector<T>&	operator=(CudaVector<T>&&);
		CudaVector<T>&	operator=(const std::vector<T>&);

		// Insert - Remove
		void			InsertEnd(const T& hostData);
		void			RemoveEnd();
		void			Assign(size_t index, const T& hostData);
		void			Assign(size_t index, const T& hostData, cudaStream_t stream);
		void			Assign(size_t index, size_t dataLength, const T* hostData);
		void			Memset(int value, size_t offset, size_t count);

		// Size Related
		void			Reserve(size_t newSize);
		void			Resize(size_t newSize);
		void			Clear();

		T*				Data();
		const T*		Data() const;
		
		size_t			Size() const;

		// Debug
		template<class O>
		void			DumpToFile(const char* fName) const;

		template<class O>
		void			DumpToFile(const char* fName,
								   size_t offset,
								   size_t count) const;
};
#include "CudaVector.hpp"
#endif //__CUDAVECTOR_H__