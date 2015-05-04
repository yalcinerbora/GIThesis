#include "CMatrix.cuh"

__device__ float4 MultMatrix(const float4&, const CMatrix4x4&)
{
	return { 0.0f, 0.0f, 0.0f, 0.0f };
}

__device__ CMatrix4x4 MultMatrix(const CMatrix4x4&, const CMatrix4x4&)
{
	return CMatrix4x4
	{{
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f },
		{ 0.0f, 0.0f, 0.0f, 0.0f }
	}};
}

__device__ float3 MultMatrix(const float3&, const CMatrix3x3&)
{
	return { 0.0f, 0.0f, 0.0f};
}

__device__ CMatrix3x3 MultMatrix(const CMatrix3x3&, const CMatrix3x3&)
{
	return CMatrix3x3
	{{
		{ 0.0f, 0.0f, 0.0f},
		{ 0.0f, 0.0f, 0.0f},
		{ 0.0f, 0.0f, 0.0f}
	}};
}

__device__ void MultMatrixSelf(float4&, const CMatrix4x4&)
{

}

__device__ void MultMatrixSelf(CMatrix4x4&, const CMatrix4x4&)
{

}

__device__ void MultMatrixSelf(float3&, const CMatrix3x3&)
{

}

__device__ void MultMatrixSelf(CMatrix3x3&, const CMatrix3x3&)
{

}