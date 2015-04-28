#include "IEMatrix3x3.h"
#include "IEVector3.h"
#include "IEVector4.h"
#include "IEQuaternion.h"
#include "IEMath.h"

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

IEVector3 IEMatrix3x3::operator*(const IEVector3& vector) const
{
	return IEVector3(v[0] * vector.getX() + v[3] * vector.getY() + v[6] * vector.getZ(),	// X
					 v[1] * vector.getX() + v[4] * vector.getY() + v[7] * vector.getZ(),	// Y
					 v[2] * vector.getX() + v[5] * vector.getY() + v[8] * vector.getZ());	// Z;	
}

IEVector4 IEMatrix3x3::operator*(const IEVector4& vector) const
{
	// Multing as if there is no translation component on a 4x4 matrix
	return IEVector4(v[0] * vector.getX() + v[3] * vector.getY() + v[6] * vector.getZ(),	// X
					 v[1] * vector.getX() + v[4] * vector.getY() + v[7] * vector.getZ(),	// Y
					 v[2] * vector.getX() + v[5] * vector.getY() + v[8] * vector.getZ(),	// Z
					 vector.getW());														// W	
}

IEMatrix3x3 IEMatrix3x3::operator*(const IEMatrix3x3& matrix) const
{
	return IEMatrix3x3(	// Column 1	
					   v[0] * matrix.v[0] + v[3] * matrix.v[1] + v[6] * matrix.v[2],		// X
					   v[1] * matrix.v[0] + v[4] * matrix.v[1] + v[7] * matrix.v[2],		// Y
					   v[2] * matrix.v[0] + v[5] * matrix.v[1] + v[8] * matrix.v[2],		// Z
					   

					   // Column 2	
					   v[0] * matrix.v[3] + v[3] * matrix.v[4] + v[6] * matrix.v[5],		// X
					   v[1] * matrix.v[3] + v[4] * matrix.v[4] + v[7] * matrix.v[5],		// Y
					   v[2] * matrix.v[3] + v[5] * matrix.v[4] + v[8] * matrix.v[5],		// Z
					   

					   // Column 3	
					   v[0] * matrix.v[6] + v[3] * matrix.v[7] + v[6] * matrix.v[8],		// X
					   v[1] * matrix.v[6] + v[4] * matrix.v[7] + v[7] * matrix.v[8],		// Y
					   v[2] * matrix.v[6] + v[5] * matrix.v[7] + v[8] * matrix.v[8]);		// Z
}

IEMatrix3x3 IEMatrix3x3::operator*(float t) const
{
	return IEMatrix3x3(m11 * t, m21 * t, m31 * t,
					   m12 * t, m22 * t, m32 * t,
					   m13 * t, m23 * t, m33 * t);
}

IEMatrix3x3 IEMatrix3x3::operator+(const IEMatrix3x3& matrix) const
{
	return IEMatrix3x3(m11 + matrix.m11, m21 + matrix.m21, m31 + matrix.m31,
					   m12 + matrix.m12, m22 + matrix.m22, m32 + matrix.m32,
					   m13 + matrix.m13, m23 + matrix.m23, m33 + matrix.m33);
}

IEMatrix3x3 IEMatrix3x3::operator-(const IEMatrix3x3& matrix) const
{
	return IEMatrix3x3(m11 - matrix.m11, m21 - matrix.m21, m31 - matrix.m31,
					   m12 - matrix.m12, m22 - matrix.m22, m32 - matrix.m32,
					   m13 - matrix.m13, m23 - matrix.m23, m33 - matrix.m33);
}

IEMatrix3x3 IEMatrix3x3::operator/(float t) const
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	return IEMatrix3x3(m11 * tinv, m21 * tinv, m31 * tinv, 
					   m12 * tinv, m22 * tinv, m32 * tinv, 
					   m13 * tinv, m23 * tinv, m33 * tinv);
}

void IEMatrix3x3::operator*=(const IEMatrix3x3& matrix)
{
	float copyData[9];
	std::copy(v, v + 9, copyData);

	const float* rightData = matrix.v;
	if(this == &matrix)
		rightData = copyData;

	// Column 1
	v[0] = copyData[0] * rightData[0] + copyData[3] * rightData[1] + copyData[6] * rightData[2]; 		// X
	v[1] = copyData[1] * rightData[0] + copyData[4] * rightData[1] + copyData[7] * rightData[2]; 		// Y
	v[2] = copyData[2] * rightData[0] + copyData[5] * rightData[1] + copyData[8] * rightData[2];		// Z

	// Column 2	
	v[3] = copyData[0] * rightData[3] + copyData[4] * rightData[4] + copyData[6] * rightData[5]; 		// X
	v[4] = copyData[1] * rightData[3] + copyData[4] * rightData[4] + copyData[7] * rightData[5]; 		// Y
	v[5] = copyData[2] * rightData[3] + copyData[5] * rightData[4] + copyData[8] * rightData[5];		// Z

	// Column 3	
	v[6] = copyData[0] * rightData[6] + copyData[3] * rightData[7] + copyData[6] * rightData[8];		// X
	v[7] = copyData[1] * rightData[6] + copyData[4] * rightData[7] + copyData[7] * rightData[8];		// Y
	v[8] = copyData[2] * rightData[6] + copyData[5] * rightData[7] + copyData[8] * rightData[8];		// Z
}

void IEMatrix3x3::operator*=(float t)
{
	for(int i = 0; i < 9; i++)
	{
		v[i] *= t;
	}
}

void IEMatrix3x3::operator+=(const IEMatrix3x3& matrix)
{
	for(int i = 0; i < 9; i++)
	{
		v[i] += matrix.v[i];
	}
}

void IEMatrix3x3::operator-=(const IEMatrix3x3& matrix)
{
	for(int i = 0; i < 9; i++)
	{
		v[i] -= matrix.v[i];
	}
}

void IEMatrix3x3::operator/=(float t)
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	for(int i = 0; i < 9; i++)
	{
		v[i] *= tinv;
	}
}

bool IEMatrix3x3::operator==(const IEMatrix3x3& matrix) const
{
	if(std::equal(v, v + 9, matrix.v))
		return true;
	else
		return false;
}

bool IEMatrix3x3::operator!=(const IEMatrix3x3& matrix) const
{
	if(std::equal(v, v + 9, matrix.v))
		return false;
	else
		return true;
}

float IEMatrix3x3::Determinant() const
{
	// Det1
	float det1 = v[0] * (v[4] * v[8] + v[7] * v[5]);
	float det2 = v[3] * (v[1] * v[8] + v[7] * v[2]);
	float det3 = v[6] * (v[1] * v[5] + v[4] * v[2]);

	return det1 - det2 + det3;
}

IEMatrix3x3 IEMatrix3x3::Inverse() const
{
	float m11 = v[4] * v[8] - v[7] * v[5];
	float m12 = -(v[1] * v[8] - v[7] * v[2]);
	float m13 = v[1] * v[5] - v[4] * v[2];

	float m21 = -(v[3] * v[8] - v[6] * v[5]);
	float m22 = v[0] * v[8] - v[6] * v[2];
	float m23 = -(v[0] * v[5] - v[3] * v[2]);

	float m31 = v[3] * v[7] - v[6] * v[4];
	float m32 = -(v[0] * v[7] - v[6] * v[1]);
	float m33 = v[0] * v[4] - v[3] * v[1];

	float det = v[0] * m11 + v[3] * m12 + v[6] * m13;
	if(det == 0.0f) {return *this;}
	float detInv = 1 / det;
	return detInv * IEMatrix3x3(m11, m12, m13,
								m21, m22, m23,
								m31, m32, m33);
}

IEMatrix3x3& IEMatrix3x3::InverseSelf()
{
	float inv[9], det;

	inv[0] = v[4] * v[8] - v[7] * v[5];
	inv[3] = -(v[1] * v[8] - v[7] * v[2]);
	inv[6] = v[1] * v[5] - v[4] * v[2];

	inv[1] = -(v[3] * v[8] - v[6] * v[5]);
	inv[4] = v[0] * v[8] - v[6] * v[2];
	inv[7] = -(v[0] * v[5] - v[3] * v[2]);

	inv[2] = v[3] * v[7] - v[6] * v[4];
	inv[5] = -(v[0] * v[7] - v[6] * v[1]);
	inv[8] = v[0] * v[4] - v[3] * v[1];

	det = v[0] * m11 + v[3] * m12 + v[6] * m13;
	if(det != 0.0f)
	{
		float detInv = 1 / det;
		for(int i = 0; i < 9; i++)
		{
			v[i] = inv[i] * det;
		}
	}
	return *this;
}

IEMatrix3x3 IEMatrix3x3::Transpose() const
{
	return IEMatrix3x3(v[0], v[3], v[6],
					   v[1], v[4], v[7],
					   v[2], v[5], v[8]);
}

IEMatrix3x3& IEMatrix3x3::TransposeSelf()
{
	for(int z = 0; z < 3; z++)
		for(int i = z + 1; i < 3; i++)
		{
			float swap = v[z * 3 + i];
			v[z * 3 + i] = v[i * 3 + z];
			v[i * 3 + z] = swap;
		}
	return *this;
}

// Vector Transformation Matrix Creation and Projection Matrix Creation
// All of these operations applies on to the current matrix
IEMatrix3x3 IEMatrix3x3::Rotate(float angle, const IEVector3& vector)
{
	//	r		r		r
	//	r		r		r
	//	r		r		r
	float tmp1, tmp2;

	IEVector3 normalizedVector = vector.Normalize();
	float cosAngle = IEMath::CosF(angle);
	float sinAngle = IEMath::SinF(angle);
	float t = 1.0f - cosAngle;

	tmp1 = normalizedVector.getX() * normalizedVector.getY() * t;
	tmp2 = normalizedVector.getZ() * sinAngle;
	float m21 = tmp1 + tmp2;
	float m12 = tmp1 - tmp2;

	tmp1 = normalizedVector.getX() * normalizedVector.getZ() * t;
	tmp2 = normalizedVector.getY() * sinAngle;
	float m31 = tmp1 - tmp2;
	float m13 = tmp1 + tmp2;

	tmp1 = normalizedVector.getY() * normalizedVector.getZ() * t;
	tmp2 = normalizedVector.getX() * sinAngle;
	float m32 = tmp1 + tmp2;
	float m23 = tmp1 - tmp2;

	float m11 = cosAngle + normalizedVector.getX() * normalizedVector.getX() * t;
	float m22 = cosAngle + normalizedVector.getY() * normalizedVector.getY() * t;
	float m33 = cosAngle + normalizedVector.getZ() * normalizedVector.getZ() * t;

	return IEMatrix3x3(m11, m21, m31,
					   m12, m22, m32,
					   m13, m23, m33);
}

IEMatrix3x3 IEMatrix3x3::Rotate(const IEQuaternion& quat)
{
	IEMatrix3x3 result;
	IEQuaternion normQuat(quat.Normalize());

	float xx = normQuat.getX() * normQuat.getX();
	float xy = normQuat.getX() * normQuat.getY();
	float xz = normQuat.getX() * normQuat.getZ();
	float xw = normQuat.getX() * normQuat.getW();
	float yy = normQuat.getY() * normQuat.getY();
	float yz = normQuat.getY() * normQuat.getZ();
	float yw = normQuat.getY() * normQuat.getW();
	float zz = normQuat.getZ() * normQuat.getZ();
	float zw = normQuat.getZ() * normQuat.getW();

	result.v[0] = (1.0f - (2.0f * (yy + zz)));
	result.v[3] = (2.0f * (xy - zw));
	result.v[6] = (2.0f * (xz + yw));
	
	result.v[1] = (2.0f * (xy + zw));
	result.v[4] = (1.0f - (2.0f * (xx + zz)));
	result.v[7] = (2.0f * (yz - xw));
	
	result.v[3] = (2.0f * (xz - yw));
	result.v[5] = (2.0f * (yz + xw));
	result.v[8] = (1.0f - (2.0f * (xx + yy)));
	return result;
}

// Left Scalar operators
IEMatrix3x3 operator*(float scalar, const IEMatrix3x3& matrix)
{
	return matrix * scalar;
}