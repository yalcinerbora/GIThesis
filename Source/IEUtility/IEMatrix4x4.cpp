#include "IEMath.h"
#include "IEMatrix4x4.h"
#include "IEVector3.h"
#include "IEVector4.h"
#include "IEQuaternion.h"

// Constants
const IEMatrix4x4 IEMatrix4x4::IdentityMatrix = IEMatrix4x4();
const IEMatrix4x4 IEMatrix4x4::ZeroMatrix = IEMatrix4x4(0.0f, 0.0f, 0.0f, 0.0f,
														0.0f, 0.0f, 0.0f, 0.0f,
														0.0f, 0.0f, 0.0f, 0.0f,
														0.0f, 0.0f, 0.0f, 0.0f);

IEMatrix4x4::IEMatrix4x4() : m11(1.0f), m21(0.0f), m31(0.0f), m41(0.0f), 
								m12(0.0f), m22(1.0f), m32(0.0f), m42(0.0f), 
								m13(0.0f), m23(0.0f), m33(1.0f), m43(0.0f), 
								m14(0.0f), m24(0.0f), m34(0.0f), m44(1.0f)
{}

IEMatrix4x4::IEMatrix4x4(float m11, float m21, float m31, float m41,
							float m12, float m22, float m32, float m42,
							float m13, float m23, float m33, float m43,
							float m14, float m24, float m34, float m44)	: m11(m11), m21(m21), m31(m31), m41(m41), 
																			m12(m12), m22(m22), m32(m32), m42(m42), 
																			m13(m13), m23(m23), m33(m33), m43(m43), 
																			m14(m14), m24(m24), m34(m34), m44(m44)
{}

IEMatrix4x4::IEMatrix4x4(float v[]) : m11(v[0]), m21(v[1]), m31(v[2]), m41(v[3]), 
										m12(v[4]), m22(v[5]), m32(v[6]), m42(v[7]), 
										m13(v[8]), m23(v[9]), m33(v[10]), m43(v[11]), 
										m14(v[12]), m24(v[13]), m34(v[14]), m44(v[15])
{}

IEVector4 IEMatrix4x4::operator*(const IEVector3& vector) const
{
	// Multing as if w component is 1.0f
	return IEVector4(	v[0] * vector.getX() + v[4] * vector.getY() + v[8] * vector.getZ() + v[12],		// X
						v[1] * vector.getX() + v[5] * vector.getY() + v[9] * vector.getZ()  + v[13],	// Y
						v[2] * vector.getX() + v[6] * vector.getY() + v[10] * vector.getZ()  + v[14],	// Z
						v[3] * vector.getX() + v[7] * vector.getY() + v[11] * vector.getZ()  + v[15]);	// W;	
}

IEVector4 IEMatrix4x4::operator*(const IEVector4& vector) const
{
	return IEVector4(	v[0] * vector.getX() + v[4] * vector.getY() + v[8] * vector.getZ() + v[12] * vector.getW(),		// X
						v[1] * vector.getX() + v[5] * vector.getY() + v[9] * vector.getZ()  + v[13] * vector.getW(),	// Y
						v[2] * vector.getX() + v[6] * vector.getY() + v[10] * vector.getZ()  + v[14] * vector.getW(),	// Z
						v[3] * vector.getX() + v[7] * vector.getY() + v[11] * vector.getZ()  + v[15] * vector.getW());	// W
}

IEMatrix4x4 IEMatrix4x4::operator*(const IEMatrix4x4& matrix) const
{
	return IEMatrix4x4(	// Column 1	
						v[0] * matrix.v[0] + v[4] * matrix.v[1] + v[8] * matrix.v[2] + v[12] * matrix.v[3],			// X
						v[1] * matrix.v[0] + v[5] * matrix.v[1] + v[9] * matrix.v[2] + v[13] * matrix.v[3],			// Y
						v[2] * matrix.v[0] + v[6] * matrix.v[1] + v[10] * matrix.v[2] + v[14] * matrix.v[3],		// Z
						v[3] * matrix.v[0] + v[7] * matrix.v[1] + v[11] * matrix.v[2] + v[15] * matrix.v[3],		// W

						// Column 2	
						v[0] * matrix.v[4] + v[4] * matrix.v[5] + v[8] * matrix.v[6] + v[12] * matrix.v[7],			// X
						v[1] * matrix.v[4] + v[5] * matrix.v[5] + v[9] * matrix.v[6] + v[13] * matrix.v[7],			// Y
						v[2] * matrix.v[4] + v[6] * matrix.v[5] + v[10] * matrix.v[6] + v[14] * matrix.v[7],		// Z
						v[3] * matrix.v[4] + v[7] * matrix.v[5] + v[11] * matrix.v[6] + v[15] * matrix.v[7],		// W

						// Column 3	
						v[0] * matrix.v[8] + v[4] * matrix.v[9] + v[8] * matrix.v[10] + v[12] * matrix.v[11],		// X
						v[1] * matrix.v[8] + v[5] * matrix.v[9] + v[9] * matrix.v[10] + v[13] * matrix.v[11],		// Y
						v[2] * matrix.v[8] + v[6] * matrix.v[9] + v[10] * matrix.v[10] + v[14] * matrix.v[11],		// Z
						v[3] * matrix.v[8] + v[7] * matrix.v[9] + v[11] * matrix.v[10] + v[15] * matrix.v[11],		// W

						// Column 4	
						v[0] * matrix.v[12] + v[4] * matrix.v[13] + v[8] * matrix.v[14] + v[12] * matrix.v[15],		// X
						v[1] * matrix.v[12] + v[5] * matrix.v[13] + v[9] * matrix.v[14] + v[13] * matrix.v[15],		// Y
						v[2] * matrix.v[12] + v[6] * matrix.v[13] + v[10] * matrix.v[14] + v[14] * matrix.v[15],	// Z
						v[3] * matrix.v[12] + v[7] * matrix.v[13] + v[11] * matrix.v[14] + v[15] * matrix.v[15]		// W
					);
}

IEMatrix4x4 IEMatrix4x4::operator*(float t) const
{
	return IEMatrix4x4(m11 * t, m21 * t, m31 * t, m41 * t,
						m12 * t, m22 * t, m32 * t, m42 * t,
						m13 * t, m23 * t, m33 * t, m43 * t,
						m14 * t, m24 * t, m34 * t, m44 * t);
}

IEMatrix4x4 IEMatrix4x4::operator+(const IEMatrix4x4& matrix) const
{
	return IEMatrix4x4(m11 + matrix.m11, m21 + matrix.m21, m31 + matrix.m31, m41 + matrix.m41,
						m12 + matrix.m12, m22 + matrix.m22, m32 + matrix.m32, m42 + matrix.m42,
						m13 + matrix.m13, m23 + matrix.m23, m33 + matrix.m33, m43 + matrix.m43,
						m14 + matrix.m14, m24 + matrix.m24, m34 + matrix.m34, m44 + matrix.m44);
}

IEMatrix4x4 IEMatrix4x4::operator-(const IEMatrix4x4& matrix) const
{
	return IEMatrix4x4(m11 - matrix.m11, m21 - matrix.m21, m31 - matrix.m31, m41 - matrix.m41,
						m12 - matrix.m12, m22 - matrix.m22, m32 - matrix.m32, m42 - matrix.m42,
						m13 - matrix.m13, m23 - matrix.m23, m33 - matrix.m33, m43 - matrix.m43,
						m14 - matrix.m14, m24 - matrix.m24, m34 - matrix.m34, m44 - matrix.m44);
}

IEMatrix4x4 IEMatrix4x4::operator/(float t) const
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	return IEMatrix4x4(m11 * tinv, m21 * tinv, m31 * tinv, m41 * tinv,
						m12 * tinv, m22 * tinv, m32 * tinv, m42 * tinv,
						m13 * tinv, m23 * tinv, m33 * tinv, m43 * tinv,
						m14 * tinv, m24 * tinv, m34 * tinv, m44 * tinv);
}

void IEMatrix4x4::operator*=(const IEMatrix4x4& matrix)
{
	float copyData[16];
	std::copy(v, v + 16, copyData);

	const float* rightData = matrix.v;
	if(this == &matrix)
		rightData = copyData;

	// Column 1
	v[0] = copyData[0] * rightData[0] + copyData[4] * rightData[1] + copyData[8] * rightData[2] + copyData[12] * rightData[3];			// X
	v[1] = copyData[1] * rightData[0] + copyData[5] * rightData[1] + copyData[9] * rightData[2] + copyData[13] * rightData[3];			// Y
	v[2] = copyData[2] * rightData[0] + copyData[6] * rightData[1] + copyData[10] * rightData[2] + copyData[14] * rightData[3];			// Z
	v[3] = copyData[3] * rightData[0] + copyData[7] * rightData[1] + copyData[11] * rightData[2] + copyData[15] * rightData[3];			// W

	// Column 2	
	v[4] = copyData[0] * rightData[4] + copyData[4] * rightData[5] + copyData[8] * rightData[6] + copyData[12] * rightData[7];			// X
	v[5] = copyData[1] * rightData[4] + copyData[5] * rightData[5] + copyData[9] * rightData[6] + copyData[13] * rightData[7];			// Y
	v[6] = copyData[2] * rightData[4] + copyData[6] * rightData[5] + copyData[10] * rightData[6] + copyData[14] * rightData[7];			// Z
	v[7] = copyData[3] * rightData[4] + copyData[7] * rightData[5] + copyData[11] * rightData[6] + copyData[15] * rightData[7];			// W

	// Column 3	
	v[8] = copyData[0] * rightData[8] + copyData[4] * rightData[9] + copyData[8] * rightData[10] + copyData[12] * rightData[11];		// X
	v[9] = copyData[1] * rightData[8] + copyData[5] * rightData[9] + copyData[9] * rightData[10] + copyData[13] * rightData[11];		// Y
	v[10] = copyData[2] * rightData[8] + copyData[6] * rightData[9] + copyData[10] * rightData[10] + copyData[14] * rightData[11];		// Z
	v[11] = copyData[3] * rightData[8] + copyData[7] * rightData[9] + copyData[11] * rightData[10] + copyData[15] * rightData[11];		// W

	// Column 4	
	v[12] = copyData[0] * rightData[12] + copyData[4] * rightData[13] + copyData[8] * rightData[14] + copyData[12] * rightData[15];		// X
	v[13] = copyData[1] * rightData[12] + copyData[5] * rightData[13] + copyData[9] * rightData[14] + copyData[13] * rightData[15];		// Y
	v[14] = copyData[2] * rightData[12] + copyData[6] * rightData[13] + copyData[10] * rightData[14] + copyData[14] * rightData[15];	// Z
	v[15] = copyData[3] * rightData[12] + copyData[7] * rightData[13] + copyData[11] * rightData[14] + copyData[15] * rightData[15];	// W
}

void IEMatrix4x4::operator*=(float t)
{
	for(int i = 0; i < 16; i++)
	{
		v[i] *= t;
	}
}

void IEMatrix4x4::operator+=(const IEMatrix4x4& matrix)
{
	for(int i = 0; i < 16; i++)
	{
		v[i] += matrix.v[i];
	}
}

void IEMatrix4x4::operator-=(const IEMatrix4x4& matrix)
{
	for(int i = 0; i < 16; i++)
	{
		v[i] -= matrix.v[i];
	}  
}

void IEMatrix4x4::operator/=(float t)
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	for(int i = 0; i < 16; i++)
	{
		v[i] *= tinv;
	}   
}

bool IEMatrix4x4::operator==(const IEMatrix4x4& matrix) const
{
	if(std::equal(v, v + 16, matrix.v))
		return true;
	else
		return false;
}

bool IEMatrix4x4::operator!=(const IEMatrix4x4& matrix) const
{
	if(std::equal(v, v + 16, matrix.v))
		return false;
	else
		return true;
}

float IEMatrix4x4::Determinant() const
{

	// Det1
	float det1 = v[0] * (	  v[5] * v[10] * v[15] 
							+ v[9] * v[14] * v[7] 
							+ v[6] * v[11] * v[13]
							- v[13] * v[10] * v[7]
							- v[9] * v[6] * v[15]
							- v[5] * v[14] * v[11]);

	float det2 = v[4] * (     v[1] * v[10] * v[15] 
							+ v[9] * v[14] * v[3] 
							+ v[2] * v[11] * v[13]
							- v[3] * v[10] * v[13]
							- v[2] * v[9] * v[15]
							- v[1] * v[11] * v[14]);

	float det3 =  v[8] * (    v[1] * v[6] * v[15] 
							+ v[5] * v[14] * v[3] 
							+ v[2] * v[7] * v[13]
							- v[3] * v[6] * v[13]
							- v[2] * v[5] * v[15]
							- v[14] * v[7] * v[1]);

	float det4 = v[12] * (    v[1] * v[6] * v[11] 
							+ v[5] * v[10] * v[3] 
							+ v[2] * v[7] * v[9]
							- v[9] * v[6] * v[3]
							- v[2] * v[5] * v[11]
							- v[1] * v[10] * v[7]);

	return det1 - det2 + det3 - det4;
}

IEMatrix4x4 IEMatrix4x4::Inverse() const
{
	IEMatrix4x4 result;
	float inv[16], det;

	// 16 determinants and one main determinant
	inv[0] = v[5]  * v[10] * v[15] - 
             v[5]  * v[11] * v[14] - 
             v[9]  * v[6]  * v[15] + 
             v[9]  * v[7]  * v[14] +
             v[13] * v[6]  * v[11] - 
             v[13] * v[7]  * v[10];

    inv[4] = -v[4]  * v[10] * v[15] + 
              v[4]  * v[11] * v[14] + 
              v[8]  * v[6]  * v[15] - 
              v[8]  * v[7]  * v[14] - 
              v[12] * v[6]  * v[11] + 
              v[12] * v[7]  * v[10];

    inv[8] = v[4]  * v[9] * v[15] - 
             v[4]  * v[11] * v[13] - 
             v[8]  * v[5] * v[15] + 
             v[8]  * v[7] * v[13] + 
             v[12] * v[5] * v[11] - 
             v[12] * v[7] * v[9];

    inv[12] = -v[4]  * v[9] * v[14] + 
               v[4]  * v[10] * v[13] +
               v[8]  * v[5] * v[14] - 
               v[8]  * v[6] * v[13] - 
               v[12] * v[5] * v[10] + 
               v[12] * v[6] * v[9];

    inv[1] = -v[1]  * v[10] * v[15] + 
              v[1]  * v[11] * v[14] + 
              v[9]  * v[2] * v[15] - 
              v[9]  * v[3] * v[14] - 
              v[13] * v[2] * v[11] + 
              v[13] * v[3] * v[10];

    inv[5] = v[0]  * v[10] * v[15] - 
             v[0]  * v[11] * v[14] - 
             v[8]  * v[2] * v[15] + 
             v[8]  * v[3] * v[14] + 
             v[12] * v[2] * v[11] - 
             v[12] * v[3] * v[10];

    inv[9] = -v[0]  * v[9] * v[15] + 
              v[0]  * v[11] * v[13] + 
              v[8]  * v[1] * v[15] - 
              v[8]  * v[3] * v[13] - 
              v[12] * v[1] * v[11] + 
              v[12] * v[3] * v[9];

    inv[13] = v[0]  * v[9] * v[14] - 
              v[0]  * v[10] * v[13] - 
              v[8]  * v[1] * v[14] + 
              v[8]  * v[2] * v[13] + 
              v[12] * v[1] * v[10] - 
              v[12] * v[2] * v[9];

    inv[2] = v[1]  * v[6] * v[15] - 
             v[1]  * v[7] * v[14] - 
             v[5]  * v[2] * v[15] + 
             v[5]  * v[3] * v[14] + 
             v[13] * v[2] * v[7] - 
             v[13] * v[3] * v[6];

    inv[6] = -v[0]  * v[6] * v[15] + 
              v[0]  * v[7] * v[14] + 
              v[4]  * v[2] * v[15] - 
              v[4]  * v[3] * v[14] - 
              v[12] * v[2] * v[7] + 
              v[12] * v[3] * v[6];

    inv[10] = v[0]  * v[5] * v[15] - 
              v[0]  * v[7] * v[13] - 
              v[4]  * v[1] * v[15] + 
              v[4]  * v[3] * v[13] + 
              v[12] * v[1] * v[7] - 
              v[12] * v[3] * v[5];

    inv[14] = -v[0]  * v[5] * v[14] + 
               v[0]  * v[6] * v[13] + 
               v[4]  * v[1] * v[14] - 
               v[4]  * v[2] * v[13] - 
               v[12] * v[1] * v[6] + 
               v[12] * v[2] * v[5];

    inv[3] = -v[1] * v[6] * v[11] + 
              v[1] * v[7] * v[10] + 
              v[5] * v[2] * v[11] - 
              v[5] * v[3] * v[10] - 
              v[9] * v[2] * v[7] + 
              v[9] * v[3] * v[6];

    inv[7] = v[0] * v[6] * v[11] - 
             v[0] * v[7] * v[10] - 
             v[4] * v[2] * v[11] + 
             v[4] * v[3] * v[10] + 
             v[8] * v[2] * v[7] - 
             v[8] * v[3] * v[6];

    inv[11] = -v[0] * v[5] * v[11] + 
               v[0] * v[7] * v[9] + 
               v[4] * v[1] * v[11] - 
               v[4] * v[3] * v[9] - 
               v[8] * v[1] * v[7] + 
               v[8] * v[3] * v[5];

    inv[15] = v[0] * v[5] * v[10] - 
              v[0] * v[6] * v[9] - 
              v[4] * v[1] * v[10] + 
              v[4] * v[2] * v[9] + 
              v[8] * v[1] * v[6] - 
              v[8] * v[2] * v[5];

    det = v[0] * inv[0] + v[1] * inv[4] + v[2] * inv[8] + v[3] * inv[12];

    if (det == 0.0f)
        return result;

    det = 1.0f / det;
    for (int i = 0; i < 16; i++)
        result.v[i] = inv[i] * det;

	return result;
}

IEMatrix4x4& IEMatrix4x4::InverseSelf()
{
	float inv[16], det;

	// 16 determinants and one main determinant
	inv[0] = v[5]  * v[10] * v[15] - 
             v[5]  * v[11] * v[14] - 
             v[9]  * v[6]  * v[15] + 
             v[9]  * v[7]  * v[14] +
             v[13] * v[6]  * v[11] - 
             v[13] * v[7]  * v[10];

    inv[4] = -v[4]  * v[10] * v[15] + 
              v[4]  * v[11] * v[14] + 
              v[8]  * v[6]  * v[15] - 
              v[8]  * v[7]  * v[14] - 
              v[12] * v[6]  * v[11] + 
              v[12] * v[7]  * v[10];

    inv[8] = v[4]  * v[9] * v[15] - 
             v[4]  * v[11] * v[13] - 
             v[8]  * v[5] * v[15] + 
             v[8]  * v[7] * v[13] + 
             v[12] * v[5] * v[11] - 
             v[12] * v[7] * v[9];

    inv[12] = -v[4]  * v[9] * v[14] + 
               v[4]  * v[10] * v[13] +
               v[8]  * v[5] * v[14] - 
               v[8]  * v[6] * v[13] - 
               v[12] * v[5] * v[10] + 
               v[12] * v[6] * v[9];

    inv[1] = -v[1]  * v[10] * v[15] + 
              v[1]  * v[11] * v[14] + 
              v[9]  * v[2] * v[15] - 
              v[9]  * v[3] * v[14] - 
              v[13] * v[2] * v[11] + 
              v[13] * v[3] * v[10];

    inv[5] = v[0]  * v[10] * v[15] - 
             v[0]  * v[11] * v[14] - 
             v[8]  * v[2] * v[15] + 
             v[8]  * v[3] * v[14] + 
             v[12] * v[2] * v[11] - 
             v[12] * v[3] * v[10];

    inv[9] = -v[0]  * v[9] * v[15] + 
              v[0]  * v[11] * v[13] + 
              v[8]  * v[1] * v[15] - 
              v[8]  * v[3] * v[13] - 
              v[12] * v[1] * v[11] + 
              v[12] * v[3] * v[9];

    inv[13] = v[0]  * v[9] * v[14] - 
              v[0]  * v[10] * v[13] - 
              v[8]  * v[1] * v[14] + 
              v[8]  * v[2] * v[13] + 
              v[12] * v[1] * v[10] - 
              v[12] * v[2] * v[9];

    inv[2] = v[1]  * v[6] * v[15] - 
             v[1]  * v[7] * v[14] - 
             v[5]  * v[2] * v[15] + 
             v[5]  * v[3] * v[14] + 
             v[13] * v[2] * v[7] - 
             v[13] * v[3] * v[6];

    inv[6] = -v[0]  * v[6] * v[15] + 
              v[0]  * v[7] * v[14] + 
              v[4]  * v[2] * v[15] - 
              v[4]  * v[3] * v[14] - 
              v[12] * v[2] * v[7] + 
              v[12] * v[3] * v[6];

    inv[10] = v[0]  * v[5] * v[15] - 
              v[0]  * v[7] * v[13] - 
              v[4]  * v[1] * v[15] + 
              v[4]  * v[3] * v[13] + 
              v[12] * v[1] * v[7] - 
              v[12] * v[3] * v[5];

    inv[14] = -v[0]  * v[5] * v[14] + 
               v[0]  * v[6] * v[13] + 
               v[4]  * v[1] * v[14] - 
               v[4]  * v[2] * v[13] - 
               v[12] * v[1] * v[6] + 
               v[12] * v[2] * v[5];

    inv[3] = -v[1] * v[6] * v[11] + 
              v[1] * v[7] * v[10] + 
              v[5] * v[2] * v[11] - 
              v[5] * v[3] * v[10] - 
              v[9] * v[2] * v[7] + 
              v[9] * v[3] * v[6];

    inv[7] = v[0] * v[6] * v[11] - 
             v[0] * v[7] * v[10] - 
             v[4] * v[2] * v[11] + 
             v[4] * v[3] * v[10] + 
             v[8] * v[2] * v[7] - 
             v[8] * v[3] * v[6];

    inv[11] = -v[0] * v[5] * v[11] + 
               v[0] * v[7] * v[9] + 
               v[4] * v[1] * v[11] - 
               v[4] * v[3] * v[9] - 
               v[8] * v[1] * v[7] + 
               v[8] * v[3] * v[5];

    inv[15] = v[0] * v[5] * v[10] - 
              v[0] * v[6] * v[9] - 
              v[4] * v[1] * v[10] + 
              v[4] * v[2] * v[9] + 
              v[8] * v[1] * v[6] - 
              v[8] * v[2] * v[5];

    det = v[0] * inv[0] + v[1] * inv[4] + v[2] * inv[8] + v[3] * inv[12];

    if (det != 0.0f)
	{
		det = 1.0f / det;
		for (int i = 0; i < 16; i++)
			v[i] = inv[i] * det;
	}
	return *this;
}

IEMatrix4x4 IEMatrix4x4::Transpose() const
{
	return IEMatrix4x4( v[0], v[4], v[8], v[12],
						v[1], v[5], v[9], v[13],
						v[2], v[6], v[10], v[14],
						v[3], v[7], v[11], v[15]);
}

IEMatrix4x4& IEMatrix4x4::TransposeSelf()
{
	for(int z = 0; z < 4; z++)
		for(int i = z + 1; i < 4; i++)
		{
			float swap = v[z * 4 + i];
			v[z * 4 + i] = v[i * 4 + z];
			v[i * 4 + z] = swap;
		}
	return *this;
}

IEMatrix4x4 IEMatrix4x4::Translate(const IEVector3& vector)
{
	//	1		0		0		tx
	//	0		1		0		ty
	//	0		0		1		tz
	//	0		0		0		1
	return IEMatrix4x4(	1.0f,			0.0f,			0.0f,		0.0f,
						0.0f,			1.0f,			0.0f,		0.0f,
						0.0f,			0.0f,			1.0f,		0.0f,
						vector.getX(), vector.getY(), vector.getZ(), 1.0f);

}

IEMatrix4x4 IEMatrix4x4::Scale(float s)
{
	//	s		0		0		0
	//	0		s		0		0
	//	0		0		s		0
	//	0		0		0		1
	return IEMatrix4x4(	s,				0.0f,			0.0f,		0.0f,
						0.0f,			s,				0.0f,		0.0f,
						0.0f,			0.0f,			s,			0.0f,
						0.0f,			0.0f,			0.0f,		1.0f
						);
}

IEMatrix4x4 IEMatrix4x4::Scale(float x, float y, float z)
{
	//	sx		0		0		0
	//	0		sy		0		0
	//	0		0		sz		0
	//	0		0		0		1
	return IEMatrix4x4(	x,				0.0f,			0.0f,		0.0f,
						0.0f,			y,				0.0f,		0.0f,
						0.0f,			0.0f,			z,			0.0f,
						0.0f,			0.0f,			0.0f,		1.0f
						);
}

IEMatrix4x4 IEMatrix4x4::Rotate(float angle, const IEVector3& vector)
{
	//	r		r		r		0
	//	r		r		r		0
	//	r		r		r		0
	//	0		0		0		1
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

	return IEMatrix4x4(	m11,			m21,			m31,		0.0f,
						m12,			m22,			m32,		0.0f,
						m13,			m23,			m33,		0.0f,
						0.0f,			0.0f,			0.0f,		1.0f
						);
}

IEMatrix4x4 IEMatrix4x4::Rotate(const IEQuaternion& quat)
{
	IEMatrix4x4 result;
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

	result.v[0]  = (1.0f - (2.0f * (yy + zz)));
	result.v[4]  =         (2.0f * (xy - zw));
	result.v[8]  =         (2.0f * (xz + yw));
	result.v[12] = 0.0f;
	result.v[1]  =         (2.0f * (xy + zw));
	result.v[5]  = (1.0f - (2.0f * (xx + zz)));
	result.v[9]  =         (2.0f * (yz - xw));
	result.v[13] = 0.0f;
	result.v[2]  =         (2.0f * (xz - yw));
	result.v[6]  =         (2.0f * (yz + xw));
	result.v[10] = (1.0f - (2.0f * (xx + yy)));
	result.v[14] = 0.0f;
	result.v[3]	 = 0.0f;
	result.v[7]  = 0.0f;
	result.v[11] = 0.0f;
	result.v[15] = 1.0f;

	return result;
}

IEMatrix4x4 IEMatrix4x4::Perspective(float fovXDegrees, float aspectRatio,
										float nearPlane, float farPlane)
{
	//	p		0		0		0
	//	0		p		0		0
	//	0		0		p		-1
	//	0		0		p		0
	float fovXRadians = IEMath::ToRadians(fovXDegrees);
	float f = 1.0f / IEMath::TanF(fovXRadians * 0.5f);
	float m33 = (farPlane + nearPlane) / (nearPlane - farPlane);
	float m34 = (2 * farPlane * nearPlane) /  (nearPlane - farPlane);
	//float m33 = farPlane / (nearPlane - farPlane);
	//float m34 = (nearPlane * farPlane) / (nearPlane - farPlane);

	return IEMatrix4x4(	f,			0.0f,				0.0f,		0.0f,
						0.0f,		f * aspectRatio,	0.0f,		0.0f,
						0.0f,		0.0f,				m33,		-1.0f,
						0.0f,		0.0f,				m34,		0.0f
						);
}

IEMatrix4x4 IEMatrix4x4::Ortogonal(float left, float right, 
									float top, float bottom,
									float nearPlane, float farPlane)
{
	//	orto	0		0		0
	//	0		orto	0		0
	//	0		0		orto	0
	//	orto	orto	orto	1
	float xt = - ((right + left) / (right - left));
	float yt = - ((top + bottom) / (top - bottom));
	float zt = ((farPlane + nearPlane) / (farPlane - nearPlane));

	return IEMatrix4x4(	2.0f / (right - left),		0.0f,					0.0f,						0.0f,
						0.0f,						2.0f / (top - bottom),	0.0f,						0.0f,
						0.0f,						0.0f,					-2.0f / (farPlane - nearPlane),	0.0f,
						xt,							yt,						zt,						1.0f
						);
}

IEMatrix4x4 IEMatrix4x4::LookAt(const IEVector3& eyePos, 
								const IEVector3& center, 
								const IEVector3& up)
{
	// Calculate Ortogonal Vectors for this rotation
	IEVector3 zAxis = (eyePos - center).NormalizeSelf();
	IEVector3 xAxis = up.CrossProduct(zAxis).NormalizeSelf();
	IEVector3 yAxis = zAxis.CrossProduct(xAxis).NormalizeSelf();

	// Also Add Translation part
	return IEMatrix4x4(	xAxis.getX(),				yAxis.getX(),				zAxis.getX(),				0.0f,
						xAxis.getY(),				yAxis.getY(),				zAxis.getY(),				0.0f,
						xAxis.getZ(),				yAxis.getZ(),				zAxis.getZ(),				0.0f,
						-xAxis.DotProduct(eyePos),	-yAxis.DotProduct(eyePos),	-zAxis.DotProduct(eyePos),	1.0f
						);
}

// Left Scalar operators
IEMatrix4x4 operator*(float scalar, const IEMatrix4x4& matrix)
{
	return matrix * scalar;
}
