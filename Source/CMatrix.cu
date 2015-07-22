#include "CMatrix.cuh"
#include "math_functions.h"

__device__ float4 MultMatrix(const float4& v, const CMatrix4x4& m)
{
	return
	{
		m.column[0].x * v.x + m.column[1].x * v.y + m.column[2].x * v.z + m.column[3].x * v.w,		// X
		m.column[0].y * v.x + m.column[1].y * v.y + m.column[2].y * v.z + m.column[3].y * v.w,		// Y
		m.column[0].z * v.x + m.column[1].z * v.y + m.column[2].z * v.z + m.column[3].z * v.w,		// Z
		m.column[0].w * v.x + m.column[1].w * v.y + m.column[2].w * v.z + m.column[3].w * v.w		// W
	};
}

__device__ CMatrix4x4 MultMatrix(const CMatrix4x4& m1, const CMatrix4x4& m2)
{
	return
	{{
		// Column 1	
		{ m1.column[0].x * m2.column[0].x + m1.column[1].x * m2.column[0].y + m1.column[2].x * m2.column[0].z + m1.column[3].x * m2.column[0].w,	// X
		  m1.column[0].y * m2.column[0].x + m1.column[1].y * m2.column[0].y + m1.column[2].y * m2.column[0].z + m1.column[3].y * m2.column[0].w,	// Y
		  m1.column[0].z * m2.column[0].x + m1.column[1].z * m2.column[0].y + m1.column[2].z * m2.column[0].z + m1.column[3].z * m2.column[0].w,	// Z
		  m1.column[0].w * m2.column[0].x + m1.column[1].w * m2.column[0].y + m1.column[2].w * m2.column[0].z + m1.column[3].w * m2.column[0].w},	// W
		
		// Column 2	
		{ m1.column[0].x * m2.column[1].x + m1.column[1].x * m2.column[1].y + m1.column[2].x * m2.column[1].z + m1.column[3].x * m2.column[1].w,	// X
		  m1.column[0].y * m2.column[1].x + m1.column[1].y * m2.column[1].y + m1.column[2].y * m2.column[1].z + m1.column[3].y * m2.column[1].w,	// Y
		  m1.column[0].z * m2.column[1].x + m1.column[1].z * m2.column[1].y + m1.column[2].z * m2.column[1].z + m1.column[3].z * m2.column[1].w,	// Z
		  m1.column[0].w * m2.column[1].x + m1.column[1].w * m2.column[1].y + m1.column[2].w * m2.column[1].z + m1.column[3].w * m2.column[1].w},	// W

		// Column 3	
		{ m1.column[0].x * m2.column[2].x + m1.column[1].x * m2.column[2].y + m1.column[2].x * m2.column[2].z + m1.column[3].x * m2.column[2].w,	// X
		  m1.column[0].y * m2.column[2].x + m1.column[1].y * m2.column[2].y + m1.column[2].y * m2.column[2].z + m1.column[3].y * m2.column[2].w,	// Y
		  m1.column[0].z * m2.column[2].x + m1.column[1].z * m2.column[2].y + m1.column[2].z * m2.column[2].z + m1.column[3].z * m2.column[2].w,	// Z
		  m1.column[0].w * m2.column[2].x + m1.column[1].w * m2.column[2].y + m1.column[2].w * m2.column[2].z + m1.column[3].w * m2.column[2].w},	// W
	
		// Column 4	
		{ m1.column[0].x * m2.column[3].x + m1.column[1].x * m2.column[3].y + m1.column[2].x * m2.column[3].z + m1.column[3].x * m2.column[3].w,	// X
		  m1.column[0].y * m2.column[3].x + m1.column[1].y * m2.column[3].y + m1.column[2].y * m2.column[3].z + m1.column[3].y * m2.column[3].w,	// Y
		  m1.column[0].z * m2.column[3].x + m1.column[1].z * m2.column[3].y + m1.column[2].z * m2.column[3].z + m1.column[3].z * m2.column[3].w,	// Z
		  m1.column[0].w * m2.column[3].x + m1.column[1].w * m2.column[3].y + m1.column[2].w * m2.column[3].z + m1.column[3].w * m2.column[3].w}	// W
	}};
}

__device__ float3 MultMatrix(const float3& v, const CMatrix3x3& m)
{
	return 
	{ 
		m.column[0].x * v.x + m.column[1].x * v.y + m.column[2].x * v.z,		// X
		m.column[0].y * v.x + m.column[1].y * v.y + m.column[2].y * v.z,		// Y
		m.column[0].z * v.x + m.column[1].z * v.y + m.column[2].z * v.z,		// Z
	};
}

__device__ float3 MultMatrix(float3& v, const CMatrix4x4& m)
{
	return
	{
		m.column[0].x * v.x + m.column[1].x * v.y + m.column[2].x * v.z,		// X
		m.column[0].y * v.x + m.column[1].y * v.y + m.column[2].y * v.z,		// Y
		m.column[0].z * v.x + m.column[1].z * v.y + m.column[2].z * v.z,		// Z
	};
}

__device__ CMatrix3x3 MultMatrix(const CMatrix3x3& m1, const CMatrix3x3& m2)
{
	return
	{{
		// Column 1	
		{ m1.column[0].x * m2.column[0].x + m1.column[1].x * m2.column[0].y + m1.column[2].x * m2.column[0].z,	// X
		  m1.column[0].y * m2.column[0].x + m1.column[1].y * m2.column[0].y + m1.column[2].y * m2.column[0].z,	// Y
		  m1.column[0].z * m2.column[0].x + m1.column[1].z * m2.column[0].y + m1.column[2].z * m2.column[0].z},	// Z

		// Column 2	
		{ m1.column[0].x * m2.column[1].x + m1.column[1].x * m2.column[1].y + m1.column[2].x * m2.column[1].z,	// X
		  m1.column[0].y * m2.column[1].x + m1.column[1].y * m2.column[1].y + m1.column[2].y * m2.column[1].z,	// Y
		  m1.column[0].z * m2.column[1].x + m1.column[1].z * m2.column[1].y + m1.column[2].z * m2.column[1].z},	// Z

		// Column 3	
		{ m1.column[0].x * m2.column[2].x + m1.column[1].x * m2.column[2].y + m1.column[2].x * m2.column[2].z,	// X
		  m1.column[0].y * m2.column[2].x + m1.column[1].y * m2.column[2].y + m1.column[2].y * m2.column[2].z,	// Y
		  m1.column[0].z * m2.column[2].x + m1.column[1].z * m2.column[2].y + m1.column[2].z * m2.column[2].z},	// Z		
	}};
}

__device__ void MultMatrixSelf(float4& v, const CMatrix4x4& m)
{
	float4 result = MultMatrix(v, m);
	v = result;
}

__device__ void MultMatrixSelf(CMatrix4x4& m1, const CMatrix4x4& m2)
{
	CMatrix4x4 result = MultMatrix(m1, m2);
	m1.column[0] = result.column[0];
	m1.column[1] = result.column[1];
	m1.column[2] = result.column[2];
	m1.column[3] = result.column[3];
}

__device__ void MultMatrixSelf(float3& v, const CMatrix3x3& m)
{
	float3 result = MultMatrix(v, m);
	v = result;
}

__device__ void MultMatrixSelf(float3& v, const CMatrix4x4& m)
{
	float3 result = MultMatrix(v, m);
	v = result;
}

__device__ void MultMatrixSelf(CMatrix3x3& m1, const CMatrix3x3& m2)
{
	CMatrix3x3 result = MultMatrix(m1, m2);
	m1.column[0] = result.column[0];
	m1.column[1] = result.column[1];
	m1.column[2] = result.column[2];
}

__device__ float3 ExtractScaleInfo(const CMatrix4x4& m)
{
	// This is kinda hacky 
	// First it cannot determine negative scalings,
	// Second it should fail if transform matrix has shear (didnt tested tho)
	float3 result;
	result.x = sqrtf(m.column[0].x * m.column[0].x +
					 m.column[0].y * m.column[0].y +
					 m.column[0].z * m.column[0].z);

	result.y = sqrtf(m.column[1].x * m.column[1].x +
					 m.column[1].y * m.column[1].y +
					 m.column[1].z * m.column[1].z);

	result.z = sqrtf(m.column[2].x * m.column[2].x +
					 m.column[2].y * m.column[2].y +
					 m.column[2].z * m.column[2].z);
	return result;
}

__device__ float3 ExtractScaleInfo(const CMatrix3x3& m)
{
	// Same as above
	float3 result;
	result.x = sqrtf(m.column[0].x * m.column[0].x +
					 m.column[0].y * m.column[0].y +
					 m.column[0].z * m.column[0].z);
	
	result.y = sqrtf(m.column[1].x * m.column[1].x +
					 m.column[1].y * m.column[1].y +
					 m.column[1].z * m.column[1].z);

	result.z = sqrtf(m.column[2].x * m.column[2].x +
					 m.column[2].y * m.column[2].y +
					 m.column[2].z * m.column[2].z);
	return result;
}