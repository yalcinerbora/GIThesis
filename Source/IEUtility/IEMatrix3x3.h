/**

Column Major Vector Matrix

*/

#ifndef __IE_MATRIX3X3_H__
#define __IE_MATRIX3X3_H__

#include <algorithm>
#include <cassert>

#include "IEMatrix4x4.h"

class IEVector3;
class IEVector4;
class IEQuaternion;

class IEMatrix3x3
{
	private:
	union
	{
		struct
		{
			float			m11, m21, m31,
							m12, m22, m32,
							m13, m23, m33;
		};
		float				v[9];
	};

	protected:

	public:
	// Constructors & Destructor
								IEMatrix3x3();
								IEMatrix3x3(float m11, float m21, float m31,
											float m12, float m22, float m32,
											float m13, float m23, float m33);
								IEMatrix3x3(float v[]);
								IEMatrix3x3(const IEMatrix3x3&) = default;
								IEMatrix3x3(const IEMatrix4x4&);
								~IEMatrix3x3() = default;

	// Constant Matrices
	static const IEMatrix3x3	IdentityMatrix;
	static const IEMatrix3x3	ZeroMatrix;

	// Accessors
	float						operator()(int row, int column) const;
	const float*				getColumn(int column) const;
	const float*				getData() const;

	// Mutators
	void						setElement(int row, int column, float data);
	void						setColumn(int, const float[3]);
	void						setRow(int, const float[3]);
	void						setData(const float[9]);
	IEMatrix3x3&				operator=(const IEMatrix4x4&);
	IEMatrix3x3&				operator=(const IEMatrix3x3&) = default;

	// Modify		
	IEVector3					operator*(const IEVector3&) const;
	IEVector4					operator*(const IEVector4&) const;
	IEMatrix3x3					operator*(const IEMatrix3x3&) const;
	IEMatrix3x3					operator*(float) const;
	IEMatrix3x3					operator+(const IEMatrix3x3&) const;
	IEMatrix3x3					operator-(const IEMatrix3x3&) const;
	IEMatrix3x3					operator/(float) const;

	void						operator*=(const IEMatrix3x3&);
	void						operator*=(float);
	void						operator+=(const IEMatrix3x3&);
	void						operator-=(const IEMatrix3x3&);
	void						operator/=(float);

	// Logic
	bool						operator==(const IEMatrix3x3&) const;
	bool						operator!=(const IEMatrix3x3&) const;

	// Linear  Algebra
	float						Determinant() const;
	IEMatrix3x3					Inverse() const;
	IEMatrix3x3&				InverseSelf();
	IEMatrix3x3					Transpose() const;
	IEMatrix3x3&				TransposeSelf();

	// Vector Transformation Matrix Creation and Projection Matrix Creation
	// All of these operations applies on to the current matrix
	static IEMatrix3x3			Rotate(float angle, const IEVector3&);
	static IEMatrix3x3			Rotate(const IEQuaternion&);
};

// Requirements of IEMatrix3x3
static_assert(std::is_trivially_copyable<IEMatrix3x3>::value == true, "IEMatrix3x3 has to be trivially copyable");
static_assert(std::is_polymorphic<IEMatrix3x3>::value == false, "IEMatrix3x3 should not be polymorphic");
static_assert(sizeof(IEMatrix3x3) == sizeof(float) * 9, "IEMatrix3x3 size is not 36 bytes");

// Left Scalar operators
IEMatrix3x3 operator*(float, const IEMatrix3x3&);

// Inlines
inline float IEMatrix3x3::operator()(int row, int column) const
{
	assert(row >= 1 && row <= 3 && column >= 1 && column <= 3);
	return v[(column - 1) * 3 + (row - 1)];
}

inline const float* IEMatrix3x3::getColumn(int column) const
{
	assert(column >= 1 && column <= 3);
	return &v[(column - 1) * 3];
}

inline const float* IEMatrix3x3::getData() const
{
	return v;
}

inline void IEMatrix3x3::setElement(int row, int column, float data)
{
	assert(row >= 1 && row <= 3 && column >= 1 && column <= 3);
	v[(column - 1) * 3 + (row - 1)] = data;
}

inline void IEMatrix3x3::setColumn(int column, const float vector[])
{
	assert(column >= 1 && column <= 3);
	v[(column - 1) * 3] = vector[0];
	v[(column - 1) * 3 + 1] = vector[1];
	v[(column - 1) * 3 + 2] = vector[2];
	v[(column - 1) * 3 + 3] = vector[3];
}

inline void IEMatrix3x3::setRow(int row, const float vector[])
{
	assert(row >= 1 && row <= 3);
	v[(row - 1)] = vector[0];
	v[4 + (row - 1)] = vector[1];
	v[8 + (row - 1)] = vector[2];
	v[12 + (row - 1)] = vector[3];
}

inline void IEMatrix3x3::setData(const float* data)
{
	std::copy(data, data + 16, v);
}

inline IEMatrix3x3& IEMatrix3x3::operator=(const IEMatrix4x4& matrix)
{
	std::copy(matrix.getColumn(1), matrix.getColumn(1) + 3, v + 0);
	std::copy(matrix.getColumn(2), matrix.getColumn(2) + 3, v + 3);
	std::copy(matrix.getColumn(3), matrix.getColumn(3) + 3, v + 6);
	return *this;
}

//inline IEMatrix3x3& IEMatrix3x3::operator=(const IEMatrix3x3& matrix)
//{
//	std::copy(matrix.v, matrix.v + 9, v);
//	return *this;
//}
#endif //__IE_MATRIX3X3_H__