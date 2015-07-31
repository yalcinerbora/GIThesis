/**

Column Major Matrices
Only Supports mult

*/

#ifndef __CMATRIX_H__
#define __CMATRIX_H__

#include <vector_types.h>

#pragma pack(push, 1)
struct CMatrix4x4
{
	float4 column[4];
};

struct CMatrix3x3
{
	float4 column[3];
};
#pragma pack(pop)

extern __device__ float4 MultMatrix(const float4&, const CMatrix4x4&);
extern __device__ CMatrix4x4 MultMatrix(const CMatrix4x4&, const CMatrix4x4&);
extern __device__ float3 MultMatrix(const float3&, const CMatrix3x3&);
extern __device__ float3 MultMatrix(const float3&, const CMatrix4x4&);
extern __device__ CMatrix3x3 MultMatrix(const CMatrix3x3&, const CMatrix3x3&);

extern __device__ void MultMatrixSelf(float4&, const CMatrix4x4&);
extern __device__ void MultMatrixSelf(CMatrix4x4&, const CMatrix4x4&);
extern __device__ void MultMatrixSelf(float3&, const CMatrix3x3&);
extern __device__ void MultMatrixSelf(float3&, const CMatrix4x4&);
extern __device__ void MultMatrixSelf(CMatrix3x3&, const CMatrix3x3&);

extern __device__ float3 ExtractScaleInfo(const CMatrix4x4&);
extern __device__ float3 ExtractScaleInfo(const CMatrix3x3&);

#endif //__CMATRIX4X4_H__