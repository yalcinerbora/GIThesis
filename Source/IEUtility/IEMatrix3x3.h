#pragma once
/**

Column Major Vector Matrix

*/
#include <algorithm>
#include <cassert>

#include "IEMatrix4x4.h"
#include "IEVector3.h"

class IEVector3;
class IEVector4;
class IEQuaternion;

class IEMatrix3x3
{
	private:
		static constexpr int	MatrixWH = 3;

		union
		{
			struct
			{
				float			m11, m21, m31,
								m12, m22, m32,
								m13, m23, m33;
			};
			float				v[MatrixWH * MatrixWH];
		};

	protected:

	public:
	// Constructors & Destructor
								IEMatrix3x3();
								IEMatrix3x3(float m11, float m21, float m31,
											float m12, float m22, float m32,
											float m13, float m23, float m33);
								IEMatrix3x3(float v[MatrixWH * MatrixWH]);
								IEMatrix3x3(const IEVector3& c0,
											const IEVector3& c1, 
											const IEVector3& c2);
								IEMatrix3x3(const IEVector3[MatrixWH]);
								IEMatrix3x3(const IEMatrix3x3&) = default;
								IEMatrix3x3(const IEMatrix4x4&);
								~IEMatrix3x3() = default;

	// Constant Matrices
	static const IEMatrix3x3	IdentityMatrix;
	static const IEMatrix3x3	ZeroMatrix;

	// Accessor & Mutator Operators
	float&						operator()(int row, int column);
	const float&				operator()(int row, int column) const;
	float&						operator[](int);
	const float&				operator[](int) const;

	// Accessors
	const float*				getColumn(int column) const;
	IEVector3					getRow(int column) const;
	const float*				getData() const;
	
	// Mutators
	void						setColumn(int, const float[MatrixWH]);
	void						setColumn(int, const IEVector3&);
	void						setRow(int, const float[MatrixWH]);
	void						setRow(int, const IEVector3&);
	void						setData(const float[MatrixWH * MatrixWH]);
	void						setData(const IEVector3[MatrixWH]);

	IEMatrix3x3&				operator=(const IEVector3[MatrixWH]);
	IEMatrix3x3&				operator=(const IEMatrix4x4&);
	IEMatrix3x3&				operator=(const IEMatrix3x3&) = default;

	// Modify		
	IEVector3					operator*(const IEVector3&) const;
	IEVector4					operator*(const IEVector4&) const;
	IEMatrix3x3					operator*(const IEMatrix3x3&) const;
	IEMatrix3x3					operator*(float) const;
	IEMatrix3x3					operator+(const IEMatrix3x3&) const;
	IEMatrix3x3					operator-(const IEMatrix3x3&) const;
	IEMatrix3x3					operator-() const;
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
	IEMatrix3x3					Clamp(const IEMatrix3x3& min, const IEMatrix3x3& max) const;
	IEMatrix3x3					Clamp(float min, float max) const;
	IEMatrix3x3&				ClampSelf(const IEMatrix3x3& min, const IEMatrix3x3& max);
	IEMatrix3x3&				ClampSelf(float min, float max);

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
inline float& IEMatrix3x3::operator()(int row, int column)
{
	assert(row >= 0 && row < MatrixWH && 
		   column >= 0 && column < MatrixWH);
	return v[column * MatrixWH + row];
}

inline const float& IEMatrix3x3::operator()(int row, int column) const
{
	assert(row >= 0 && row < MatrixWH &&
		   column >= 0 && column < MatrixWH);
	return v[column * MatrixWH + row];
}

inline float& IEMatrix3x3::operator[](int index)
{
	assert(index >= 0 && index < MatrixWH * MatrixWH);
	return v[index];
}

inline const float& IEMatrix3x3::operator[](int index) const
{
	assert(index >= 0 && index < MatrixWH * MatrixWH);
	return v[index];
}

inline const float* IEMatrix3x3::getColumn(int column) const
{
	assert(column >= 0 && column < MatrixWH);
	return &v[column * MatrixWH];
}

inline IEVector3 IEMatrix3x3::getRow(int row) const
{
	assert(row >= 0 && row < MatrixWH);
	return
	{
		v[				 row],
		v[	  MatrixWH + row],
		v[2 * MatrixWH + row]
	};
}

inline const float* IEMatrix3x3::getData() const
{
	return v;
}

inline void IEMatrix3x3::setColumn(int column, const float vector[])
{
	assert(column >= 0 && column < MatrixWH);
	std::copy(vector, vector + MatrixWH, v + column * MatrixWH);
}

inline void IEMatrix3x3::setColumn(int column, const IEVector3& vector)
{
	assert(column >= 0 && column < MatrixWH);
	std::copy(vector.getData(), vector.getData() + MatrixWH, v + column * MatrixWH);
}

inline void IEMatrix3x3::setRow(int row, const float vector[])
{
	assert(row >= 0 && row < MatrixWH);
	v[               row] = vector[0];
	v[    MatrixWH + row] = vector[1];
	v[2 * MatrixWH + row] = vector[2];
}

inline void IEMatrix3x3::setRow(int row, const IEVector3& vector)
{
	v[               row] = vector[0];
	v[    MatrixWH + row] = vector[1];
	v[2 * MatrixWH + row] = vector[2];
}

inline void IEMatrix3x3::setData(const float* data)
{
	std::copy(data, data + MatrixWH * MatrixWH, v);
}

inline void IEMatrix3x3::setData(const IEVector3 data[])
{
	const float* dataPtr = data[0].getData();
	std::copy(dataPtr, dataPtr + MatrixWH * MatrixWH, v);
}

inline IEMatrix3x3& IEMatrix3x3::operator=(const IEVector3 data[])
{
	setData(data);
	return *this;
}

inline IEMatrix3x3& IEMatrix3x3::operator=(const IEMatrix4x4& matrix)
{
	std::copy(matrix.getData(), matrix.getData() + MatrixWH, v		         );
	std::copy(matrix.getData(), matrix.getData() + MatrixWH, v +     MatrixWH);
	std::copy(matrix.getData(), matrix.getData() + MatrixWH, v + 2 * MatrixWH);
	return *this;
}
