#pragma once
/**

Column Major Vector Matrix

*/

#include <algorithm>
#include <cassert>

#include "IEVector4.h"

class IEVector3;
class IEVector4;
class IEQuaternion;
class IEMatrix3x3;

class IEMatrix4x4
{
	private:
		static constexpr int		MatrixWH = 4;
		union
		{
			struct {float			m11, m21, m31, m41,
									m12, m22, m32, m42,
									m13, m23, m33, m43,
									m14, m24, m34, m44;};
			float					v[MatrixWH * MatrixWH];
		};

	protected:

	public:
		// Constructors & Destructor
									IEMatrix4x4();
									IEMatrix4x4(float m11, float m21, float m31, float m41,
												float m12, float m22, float m32, float m42,
												float m13, float m23, float m33, float m43,
												float m14, float m24, float m34, float m44);
									IEMatrix4x4(float v[MatrixWH * MatrixWH]);
									IEMatrix4x4(const IEVector4& c0,
												const IEVector4& c1,
												const IEVector4& c2,
												const IEVector4& c3);
									IEMatrix4x4(const IEVector4 columns[MatrixWH]);
									IEMatrix4x4(const IEMatrix3x3&);
									IEMatrix4x4(const IEMatrix4x4&) = default;
									~IEMatrix4x4() = default;

		// Constant Matrices
		static const IEMatrix4x4	IdentityMatrix;
		static const IEMatrix4x4	ZeroMatrix;

		// Accessor & Mutator Operators
		float&						operator()(int row, int column);
		const float&				operator()(int row, int column) const;
		float&						operator[](int);
		const float&				operator[](int) const;

		// Accessors
		const float*				getColumn(int column) const;
		IEVector4					getRow(int column) const;
		const float*				getData() const;

		// Mutators
		void						setColumn(int, const float[MatrixWH]);
		void						setColumn(int, const IEVector4&);
		void						setRow(int, const float[MatrixWH]);
		void						setRow(int, const IEVector4&);
		void						setData(const float[MatrixWH * MatrixWH]);
		void						setData(const IEVector4[MatrixWH]);

		// Assignemnt Operator
		IEMatrix4x4&				operator=(const IEVector4[MatrixWH]);
		IEMatrix4x4&				operator=(const IEMatrix4x4&) = default;

		// Modify		
		IEVector4					operator*(const IEVector3&) const;	
		IEVector4					operator*(const IEVector4&) const;
		IEMatrix4x4					operator*(const IEMatrix4x4&) const;
		IEMatrix4x4					operator*(float) const;
		IEMatrix4x4					operator+(const IEMatrix4x4&) const;
		IEMatrix4x4					operator-(const IEMatrix4x4&) const;
		IEMatrix4x4					operator-() const;
		IEMatrix4x4					operator/(float) const;

		void						operator*=(const IEMatrix4x4&);
		void						operator*=(float);
		void						operator+=(const IEMatrix4x4&);
		void						operator-=(const IEMatrix4x4&);
		void						operator/=(float);

		// Logic
		bool						operator==(const IEMatrix4x4&) const;
		bool						operator!=(const IEMatrix4x4&) const;

		// Linear  Algebra
		float						Determinant() const;
		IEMatrix4x4					Inverse() const;
		IEMatrix4x4&				InverseSelf();
		IEMatrix4x4					Transpose() const;
		IEMatrix4x4&				TransposeSelf();
		IEMatrix4x4					Clamp(const IEMatrix4x4& min, const IEMatrix4x4& max) const;
		IEMatrix4x4					Clamp(float min, float max) const;
		IEMatrix4x4&				ClampSelf(const IEMatrix4x4& min, const IEMatrix4x4& max);
		IEMatrix4x4&				ClampSelf(float min, float max);
		IEMatrix4x4					NormalMatrix() const;

		// Vector Transformation Matrix Creation and Projection Matrix Creation
		// All of these operations applies on to the current matrix
		static IEMatrix4x4			Translate(const IEVector3&);
		static IEMatrix4x4			Scale(float);
		static IEMatrix4x4			Scale(float x, float y, float z);
		static IEMatrix4x4			Rotate(float angle, const IEVector3&);
		static IEMatrix4x4			Rotate(const IEQuaternion&);
		static IEMatrix4x4			Perspective(float fovXRadians, float aspectRatio,
												float nearPlane, float farPlane);
		static IEMatrix4x4			Ortogonal(float left, float right, 
												float top, float bottom,
												float nearPlane, float farPlane);
		static IEMatrix4x4			Ortogonal(float width, float height,
											  float nearPlane, float farPlane);
		static IEMatrix4x4			LookAt(const IEVector3& eyePos, 
											const IEVector3& at, 
											const IEVector3& up);
		static IEVector3			ExtractScaleInfo(const IEMatrix4x4&);

};

// Requirements of IEMatrix4x4
static_assert(std::is_trivially_copyable<IEMatrix4x4>::value == true, "IEMatrix4x4 has to be trivially copyable");
static_assert(std::is_polymorphic<IEMatrix4x4>::value == false, "IEMatrix4x4 should not be polymorphic");
static_assert(sizeof(IEMatrix4x4) == sizeof(float) * 16, "IEMatrix4x4 size is not 64 bytes");

// Left Scalar operators
IEMatrix4x4 operator*(float, const IEMatrix4x4&);

// Inlines
inline float& IEMatrix4x4::operator()(int row, int column)
{
	assert(row >= 0 && row < MatrixWH && 
		   column >= 0 && column < MatrixWH);
	return v[column * MatrixWH + row];
}

inline const float& IEMatrix4x4::operator()(int row, int column) const
{
	assert(row >= 0 && row < MatrixWH &&
		   column >= 0 && column < MatrixWH);
	return v[column * MatrixWH + row];
}

inline float& IEMatrix4x4::operator[](int index)
{
	assert(index >= 0 && index < MatrixWH * MatrixWH);
	return v[index];
}
inline const float& IEMatrix4x4::operator[](int index) const
{
	assert(index >= 0 && index < MatrixWH * MatrixWH);
	return v[index];
}

inline const float* IEMatrix4x4::getColumn(int column) const
{
	assert(column >= 0 && column < MatrixWH);
	return &v[column * MatrixWH];
}

inline IEVector4 IEMatrix4x4::getRow(int row) const
{
	assert(row >= 0 && row < MatrixWH);
	return 
	{
		v[				 row],
		v[    MatrixWH + row],
		v[2 * MatrixWH + row],
		v[3 * MatrixWH + row]
	};
}

inline const float* IEMatrix4x4::getData() const
{
	return v;
}

inline void IEMatrix4x4::setColumn(int column, const float vector[])
{
	assert(column >= 0 && column <= MatrixWH);
	v[column * MatrixWH] = vector[0];
	v[column * MatrixWH + 1] = vector[1];
	v[column * MatrixWH + 2] = vector[2];
	v[column * MatrixWH + 3] = vector[3];
}

inline void IEMatrix4x4::setColumn(int column, const IEVector4& vector)
{
	assert(column >= 0 && column <= MatrixWH);
	v[column * MatrixWH] = vector[0];
	v[column * MatrixWH + 1] = vector[1];
	v[column * MatrixWH + 2] = vector[2];
	v[column * MatrixWH + 3] = vector[3];
}

inline void IEMatrix4x4::setRow(int row, const float vector[])
{ 
	assert(row >= 0 && row < MatrixWH);
	v[			     row] = vector[0];
	v[    MatrixWH + row] = vector[1];
	v[2 * MatrixWH + row] = vector[2];
	v[3 * MatrixWH + row] = vector[3];
}

inline void IEMatrix4x4::setRow(int row, const IEVector4& vector)
{
	assert(row >= 0 && row < MatrixWH);
	v[			     row] = vector[0];
	v[    MatrixWH + row] = vector[1];
	v[2 * MatrixWH + row] = vector[2];
	v[3 * MatrixWH + row] = vector[3];
}

inline void IEMatrix4x4::setData(const float* data)
{
	std::copy(data, data + MatrixWH * MatrixWH, v);
}

inline void IEMatrix4x4::setData(const IEVector4 data[])
{
	const float* dataPtr = data[0].getData();
	std::copy(dataPtr, dataPtr + MatrixWH * MatrixWH, v);
}

inline IEMatrix4x4& IEMatrix4x4::operator=(const IEVector4 data[])
{
	setData(data);
	return *this;
}

inline IEMatrix4x4 IEMatrix4x4::NormalMatrix() const
{
	return (*this).Transpose().Inverse();
}