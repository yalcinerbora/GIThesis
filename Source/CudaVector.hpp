#include "CudaVector.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <algorithm>

template<class T>
void CudaVector<T>::ExtendStorage()
{
	if(capacity == 0) capacity = CUDA_VEC_INITIAL_CAPACITY;
	size_t newCount = static_cast<size_t>(capacity * CUDA_VEC_RESIZE_FACTOR);
	T* d_newAlloc = nullptr;
	cudaMalloc<T>(&d_newAlloc, sizeof(T) * newCount);
	cudaMemcpy(d_newAlloc, d_data, sizeof(T)  * size, cudaMemcpyDeviceToDevice);
	cudaFree(d_data);
	d_data = d_newAlloc;
	capacity = newCount;
}

template<class T>
CudaVector<T>::CudaVector()
	: d_data(nullptr)
	, size(0)
	, capacity(0)
{}

template<class T>
CudaVector<T>::CudaVector(size_t count)
	: d_data(nullptr)
	, size(count)
	, capacity(((count + CUDA_VEC_INITIAL_CAPACITY - 1) / CUDA_VEC_INITIAL_CAPACITY) * CUDA_VEC_INITIAL_CAPACITY)
{
	cudaMalloc<T>(&d_data, sizeof(T) * capacity);
}

template<class T>
CudaVector<T>::CudaVector(const CudaVector<T>& cp)
	: size(cp.size)
	, capacity(cp.capacity)
{
	cudaMalloc<T>(&d_data, sizeof(T) * capacity);
	cudaMemcpy(d_data, cp.d_data, sizeof(T) * cp.size, cudaMemcpyDeviceToDevice);
}

template<class T>
CudaVector<T>::CudaVector(CudaVector<T>&& mv)
	: d_data(mv.d_data)
	, size(mv.size)
	, capacity(mv.capacity)
{
	mv.d_data = nullptr;
	mv.size = 0;
	mv.capacity = 0;
}

template<class T>
CudaVector<T>::~CudaVector()
{
	cudaFree(d_data);
	d_data = nullptr;
	size = 0;
	capacity = 0;
}

template<class T>
CudaVector<T>& CudaVector<T>::operator=(CudaVector<T>&& mv)
{
	assert(this != &mv);

	d_data = mv.d_data;
	size = mv.size;
	capacity = mv.capacity;
	
	mv.d_data = nullptr;
	mv.size = 0;
	mv.capacity = 0;
	return *this;
}

template<class T>
CudaVector<T>& CudaVector<T>::operator=(const std::vector<T>& vector)
{
	if(vector.size() > capacity)
	{
		capacity = ((vector.size() + CUDA_VEC_INITIAL_CAPACITY - 1) / CUDA_VEC_INITIAL_CAPACITY) * CUDA_VEC_INITIAL_CAPACITY;		
		T* d_newAlloc = nullptr;
		cudaMalloc<T>(&d_newAlloc, sizeof(T) * capacity);
		cudaFree(d_data);
		d_data = d_newAlloc;
	}
	cudaMemcpy(d_data, vector.data(), sizeof(T) * vector.size(), cudaMemcpyHostToDevice);
	size = vector.size();
	return *this;
}

template<class T>
void CudaVector<T>::InsertEnd(const T& hostData)
{
	if(1 > capacity - size)
	{
		ExtendStorage();
	}
	cudaMemcpy(d_data + size, &hostData, sizeof(T), cudaMemcpyHostToDevice);
	size++;
}

template<class T>
void CudaVector<T>::RemoveEnd()
{
	size--;
	std::max<size_t>(0u, size);
}

template<class T>
void CudaVector<T>::Assign(size_t index, const T& hostData)
{
	assert(index < size);
	cudaMemcpy(d_data + index, &hostData, sizeof(T), cudaMemcpyHostToDevice);
}

template<class T>
void CudaVector<T>::Assign(size_t index, size_t dataLength, const T* hostData)
{
	assert(index + datalength <= size);
	cudaMemcpy(d_data + index, sizeof(T), cudaMemcpyHostToDevice);
}

template<class T>
void CudaVector<T>::Memset(int value, size_t stride, size_t count)
{
	assert(stride + count <= size);
	cudaMemset(d_data + stride, value, count * sizeof(T));
}

template<class T>
void CudaVector<T>::Reserve(size_t newSize)
{
	if(newSize > capacity)
	{
		size_t newCount = ((newSize + InitialCapacity - 1) / InitialCapacity) * InitialCapacity;
		T* d_newAlloc = nullptr;
		cudaMalloc<T>(&d_newAlloc, sizeof(T) * newCount);
		cudaMemcpy(d_newAlloc, d_data, sizeof(T)  * size, cudaMemcpyDeviceToDevice);
		cudaFree(d_data);
		d_data = d_newAlloc;
		capacity = newCount;
	}
}

template<class T>
void CudaVector<T>::Resize(size_t newSize)
{
	Reserve(newSize);
	size = newSize;
}

template<class T>
void CudaVector<T>::Clear()
{
	size = 0;
}

template<class T>
T* CudaVector<T>::Data()
{
	return d_data;
}

template<class T>
const T* CudaVector<T>::Data() const
{
	return d_data;
}

template<class T>
size_t CudaVector<T>::Size() const
{
	return size;
}