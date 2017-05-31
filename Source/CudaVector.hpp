#include "CudaVector.cuh"
#include <cassert>
#include <algorithm>
#include "CudaInit.h"

template<class T>
void CudaVector<T>::ExtendStorage()
{
	if(capacity == 0) capacity = CUDA_VEC_INITIAL_CAPACITY;
	size_t newCount = static_cast<size_t>(capacity * CUDA_VEC_RESIZE_FACTOR);
	T* d_newAlloc = nullptr;
	CUDA_CHECK(cudaMalloc<T>(&d_newAlloc, sizeof(T) * newCount));
	CUDA_CHECK(cudaMemcpy(d_newAlloc, d_data, sizeof(T)  * size, cudaMemcpyDeviceToDevice));
	CUDA_CHECK(cudaFree(d_data));
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
	CUDA_CHECK(cudaMalloc<T>(&d_data, sizeof(T) * capacity));
}

template<class T>
CudaVector<T>::CudaVector(const CudaVector<T>& cp)
	: d_data(nullptr)
	, size(cp.size)
	, capacity(cp.capacity)
{
	CUDA_CHECK(cudaMalloc<T>(&d_data, sizeof(T) * capacity));
	CUDA_CHECK(cudaMemcpy(d_data, cp.d_data, sizeof(T) * cp.size, cudaMemcpyDeviceToDevice));
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
	CUDA_CHECK(cudaFree(d_data));
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
		CUDA_CHECK(cudaMalloc<T>(&d_newAlloc, sizeof(T) * capacity));
		CUDA_CHECK(cudaFree(d_data));
		d_data = d_newAlloc;
	}
	CUDA_CHECK((cudaMemcpy(d_data, vector.data(), sizeof(T) * vector.size(), cudaMemcpyHostToDevice)));
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
	CUDA_CHECK(cudaMemcpy(d_data + size, &hostData, sizeof(T), cudaMemcpyHostToDevice));
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
	CUDA_CHECK(cudaMemcpy(d_data + index, &hostData, sizeof(T), cudaMemcpyHostToDevice));
	//cudaMemcpy(d_data + index, &hostData, sizeof(T), cudaMemcpyHostToDevice);
}

template<class T>
void CudaVector<T>::Assign(size_t index, const T& hostData, cudaStream_t stream)
{
	assert(index < size);
	CUDA_CHECK(cudaMemcpy(d_data + index, &hostData, sizeof(T), cudaMemcpyHostToDevice));
	//cudaMemcpyAsync(d_data + index, &hostData, sizeof(T), cudaMemcpyHostToDevice, stream);
}



template<class T>
void CudaVector<T>::Assign(size_t index, size_t dataLength, const T* hostData)
{
	assert(index + dataLength <= size);
	CUDA_CHECK(cudaMemcpy(d_data + index, hostData,
						  dataLength * sizeof(T), cudaMemcpyHostToDevice));
}

template<class T>
void CudaVector<T>::Memset(int value, size_t offset, size_t count)
{
	assert(offset + count <= size);
	CUDA_CHECK(cudaMemset(d_data + offset, value, count * sizeof(T)));
}

template<class T>
void CudaVector<T>::Reserve(size_t newSize)
{
	if(newSize > capacity)
	{
		size_t newCount = ((newSize + CUDA_VEC_INITIAL_CAPACITY - 1) / CUDA_VEC_INITIAL_CAPACITY) * CUDA_VEC_INITIAL_CAPACITY;
		T* d_newAlloc = nullptr;
		CUDA_CHECK((cudaMalloc<T>(&d_newAlloc, sizeof(T) * newCount)));
		CUDA_CHECK((cudaMemcpy(d_newAlloc, d_data, sizeof(T)  * size, cudaMemcpyDeviceToDevice)));
		CUDA_CHECK(cudaFree(d_data));
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


#include <fstream>
static std::ostream& operator<< (std::ostream& ostr, const ushort2& shrt2)
{
	ostr << "{" << shrt2.x << ", " << shrt2.y << "}";
	return ostr;
}

template<class T>
template<class O>
void CudaVector<T>::DumpToFile(const char* fName) const
{
	DumpToFile<O>(fName, 0, size);
}

template<class T>
template<class O>
void CudaVector<T>::DumpToFile(const char* fName,
							   size_t offset,
							   size_t count) const
{
	static_assert(sizeof(T) % sizeof(O) == 0 ||
				  sizeof(O) % sizeof(T) == 0, "Template and output should have same sizes");

	std::vector<O> cpuData;
	cpuData.resize(count * sizeof(T) / sizeof(O));
	CUDA_CHECK(cudaMemcpy(cpuData.data(), 
						  d_data + offset, 
						  count * sizeof(T), 
						  cudaMemcpyDeviceToHost));

	std::ofstream fOut;
	fOut.open(fName);

	for(const O& data : cpuData)
	{
		fOut << "0x" << std::uppercase << std::hex << data;
		fOut << "\t\t\t" << std::nouppercase << std::dec << data;
		fOut << std::endl;
	}
		
}

//template <class T>
//template <>
//inline void CudaVector<T>::DumpToFile<unsigned char>(const char* fName,
//													 size_t offset,
//													 size_t count) const
//{
//	std::vector<unsigned char> cpuData;
//	cpuData.resize(count);
//	CUDA_CHECK(cudaMemcpy(cpuData.data(), d_data + offset, count * sizeof(unsigned char), cudaMemcpyDeviceToHost));
//	
//	std::ofstream fOut;
//	fOut.open(fName);
//
//	for(const unsigned char& data : cpuData)
//		fOut << static_cast<unsigned int>(data) << "c" <<  std::endl;
//}