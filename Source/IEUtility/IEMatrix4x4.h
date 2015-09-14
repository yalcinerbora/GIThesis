/**

Column Major Vector Matrix

*/

#ifndef __IE_MATRIX4X4_H__
#define __IE_MATRIX4X4_H__

#include <algorithm>
#include <cassert>

class IEVector3;
class IEVector4;
class IEQuaternion;
class IEMatrix3x3;

class IEMatrix4x4
{
	private:
		union
		{
			struct {float			m11, m21, m31, m41,
									m12, m22, m32, m42,
									m13, m23, m33, m43,
									m14, m24, m34, m44;};
			float					v[16];
		};

	protected:

	public:
		// Constructors & Destructor
									IEMatrix4x4();
									IEMatrix4x4(float m11, float m21, float m31, float m41,
												float m12, float m22, float m32, float m42,
												float m13, float m23, float m33, float m43,
												float m14, float m24, float m34, float m44);
									IEMatrix4x4(float v[]);
									IEMatrix4x4(const IEMatrix3x3&);
									IEMatrix4x4(const IEMatrix4x4&) = default;
									~IEMatrix4x4() = default;

		// Constant Matrices
		static const IEMatrix4x4	IdentityMatrix;
		static const IEMatrix4x4	ZeroMatrix;

		// Accessors
		float						operator()(int row, int column) const;
		const float*				getColumn(int column) const;
		const float*				getData() const;

		// Mutators
		void						setElement(int row, int column, float data);
		void						setColumn(int, const float[4]);
		void						setRow(int, const float[4]);
		void						setData(const float[16]);
		IEMatrix4x4&				operator=(const IEMatrix4x4&);

		// Modify		
		IEVector4					operator*(const IEVector3&) const;	
		IEVector4					operator*(const IEVector4&) const;
		IEMatrix4x4					operator*(const IEMatrix4x4&) const;
		IEMatrix4x4					operator*(float) const;
		IEMatrix4x4					operator+(const IEMatrix4x4&) const;
		IEMatrix4x4					operator-(const IEMatrix4x4&) const;
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

		// Vector Transformation Matrix Creation and Projection Matrix Creation
		// All of these operations applies on to the current matrix
		static IEMatrix4x4			Translate(const IEVector3&);
		static IEMatrix4x4			Scale(float);
		static IEMatrix4x4			Scale(float x, float y, float z);
		static IEMatrix4x4			Rotate(float angle, const IEVector3&);
		static IEMatrix4x4			Rotate(const IEQuaternion&);
		static IEMatrix4x4			Perspective(float fovXDegrees, float aspectRatio,
												float nearPlane, float farPlane);
		static IEMatrix4x4			Ortogonal(float left, float right, 
												float top, float bottom,
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
inline float IEMatrix4x4::operator()(int row, int column) const
{
	assert(row >= 1 && row <= 4 && column >= 1 && column <= 4);
	return v[(column - 1) * 4 + (row - 1)];
}

inline const float* IEMatrix4x4::getColumn(int column) const
{
	assert(column >= 1 && column <= 4);
	return &v[(column - 1) * 4];
}

inline const float* IEMatrix4x4::getData() const
{
	return v;
}

inline void IEMatrix4x4::setElement(int row, int column, float data)
{
	assert(row >= 1 && row <= 4 && column >= 1 && column <= 4);
	v[(column - 1) * 4 + (row - 1)] = data;
}

inline void IEMatrix4x4::setColumn(int column, const float vector[])
{
	assert(column >= 1 && column <= 4);
	v[(column - 1) * 4] = vector[0];
	v[(column - 1) * 4 + 1] = vector[1];
	v[(column - 1) * 4 + 2] = vector[2];
	v[(column - 1) * 4 + 3] = vector[3];
}

inline void IEMatrix4x4::setRow(int row, const float vector[])
{ 
	assert(row >= 1 && row <= 4);
	v[(row - 1)] = vector[0];
	v[4 + (row - 1)] = vector[1];
	v[8 + (row - 1)] = vector[2];
	v[12 + (row - 1)] = vector[3];
}

inline void IEMatrix4x4::setData(const float* data)
{
	std::copy(data, data + 16, v);
}

inline IEMatrix4x4& IEMatrix4x4::operator=(const IEMatrix4x4& matrix)
{
	std::copy(matrix.v, matrix.v + 16, v);
	return *this;
}
#endif //__IE_MATRIX4X4_H__