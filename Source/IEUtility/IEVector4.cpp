#include <algorithm>
#include <cassert>

#include "IEMath.h"
#include "IEVector4.h"
#include "IEVector3.h"

const IEVector4 IEVector4::ZeroVector = IEVector4(0.0f, 0.0f, 0.0f, 0.0f);
const IEVector4 IEVector4::XAxis = IEVector4(1.0f, 0.0f, 0.0f, 0.0f);
const IEVector4 IEVector4::YAxis = IEVector4(0.0f, 1.0f, 0.0f, 0.0f);
const IEVector4 IEVector4::ZAxis = IEVector4(0.0f, 0.0f, 1.0f, 0.0f);

IEVector4::IEVector4() : x(0.0f),
							y(0.0f),
							z(0.0f),
							w(0.0f)
{}

IEVector4::IEVector4(float xx, float yy, float zz, float ww) : x(xx),
																y(yy),
																z(zz),
																w(ww)
{}

IEVector4::IEVector4(const float v[]) : x(v[0]),
										y(v[1]),
										z(v[2]),
										w(v[3])
{}

IEVector4::IEVector4(const IEVector3& cp) : x(cp.getX()),
											y(cp.getY()),
											z(cp.getZ()),
											w(1.0f)
{}

void IEVector4::operator+=(const IEVector4& vector)
{
	x += vector.x;
	y += vector.y;
	z += vector.z;
	w += vector.w;
}

void IEVector4::operator-=(const IEVector4& vector)
{
	x -= vector.x;
	y -= vector.y;
	z -= vector.z;
	w -= vector.w;
}

void IEVector4::operator*=(const IEVector4& vector)
{
	x *= vector.x;
	y *= vector.y;
	z *= vector.z;
	w *= vector.w;
}

void IEVector4::operator*=(float t)
{
	x *= t;
	y *= t;
	z *= t;
	w *= t;
}

void IEVector4::operator/=(const IEVector4& vector)
{
	x /= vector.x;
	y /= vector.y;
	z /= vector.z;
	w /= vector.w;
}

void IEVector4::operator/=(float t)
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	x *= tinv;
	y *= tinv;
	z *= tinv;
	w *= tinv;
}

IEVector4 IEVector4::operator+(const IEVector4& vector) const
{
	return IEVector4(	x + vector.x,
						y + vector.y,
						z + vector.z,
						w + vector.w);
}

IEVector4 IEVector4::operator-(const IEVector4& vector) const
{
	return IEVector4(	x - vector.x,
						y - vector.y,
						z - vector.z,
						w - vector.w);
}

IEVector4 IEVector4::operator-() const
{
	return IEVector4(-x, -y, -z, -w);
}

IEVector4 IEVector4::operator*(const IEVector4& vector) const
{
	return IEVector4(	x * vector.x,
						y * vector.y,
						z * vector.z,
						w * vector.w);
}

IEVector4 IEVector4::operator*(float t) const
{
	return IEVector4(	x * t,
						y * t,
						z * t,
						w * t);
}

IEVector4 IEVector4::operator/(const IEVector4& vector) const
{
	assert(vector.x != 0.0f && vector.y != 0.0f && vector.z != 0.0f);
	return IEVector4(	x / vector.x,
						y / vector.y,
						z / vector.z,
						w / vector.w);
}

IEVector4 IEVector4::operator/(float t) const
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	return IEVector4(	x * tinv,
						y * tinv,
						z * tinv,
						w * tinv);
}

float IEVector4::DotProduct(const IEVector4& vector) const
{
	return  x * vector.x + y * vector.y + z * vector.z + w * vector.w;
}

float IEVector4::Length() const
{
	return sqrtf(x * x + y * y + z * z + w * w);
}

float IEVector4::LengthSqr() const
{
	return x * x + y * y + z * z + w * w;
}

IEVector4 IEVector4::Normalize() const
{
	float length = Length();
	if(length != 0.0f)
		length = 1.0f / length;

	return IEVector4(	x * length,
						y * length,
						z * length,
						w * length);
}

IEVector4& IEVector4::NormalizeSelf()
{
	float length = Length();
	if(length != 0.0f)
		length = 1.0f / length;

	x *= length;
	y *= length;
	z *= length;
	w *= length;
	return *this;
}

bool IEVector4::operator==(const IEVector4& vector) const
{
	return std::equal(v, v + 4, vector.v);
}

bool IEVector4::operator!=(const IEVector4& vector) const	
{
	return !std::equal(v, v + 4, vector.v);
}

// Left Scalar operators
IEVector4 operator*(float scalar, const IEVector4& vector)
{
	return vector * scalar;
}