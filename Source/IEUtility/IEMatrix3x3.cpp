#include "IEMatrix3x3.h"

const IEMatrix3x3 IEMatrix3x3::IdentityMatrix = { 1.0f, 0.0f, 0.0f,
												  0.0f, 1.0f, 0.0f,
												  0.0f, 0.0f, 1.0f };

IEMatrix3x3::IEMatrix3x3()
	: m11(1.0f), m21(0.0f), m31(0.0f)
	, m12(0.0f), m22(1.0f), m32(0.0f)
	, m13(0.0f), m23(0.0f), m33(1.0f)
{}

IEMatrix3x3::IEMatrix3x3(float m11, float m21, float m31,
						 float m12, float m22, float m32,
						 float m13, float m23, float m33)
	: m11(m11), m21(m21), m31(m31) 
	, m12(m12), m22(m22), m32(m32) 
	, m13(m13), m23(m23), m33(m33) 
{}

IEMatrix3x3::IEMatrix3x3(float newV[])
{
	std::copy(newV, newV + 9, v);
}

IEMatrix3x3::IEMatrix3x3(const IEMatrix4x4& matrix)
{
	std::copy(matrix.getColumn(0), matrix.getColumn(0) + 3, v + 0);
	std::copy(matrix.getColumn(1), matrix.getColumn(1) + 3, v + 3);
	std::copy(matrix.getColumn(2), matrix.getColumn(2) + 3, v + 6);
}
