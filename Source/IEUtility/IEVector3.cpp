#include <algorithm>
#include <cassert>

#include "IEMath.h"
#include "IEVector3.h"
#include "IEVector4.h"

// Constants
const IEVector3 IEVector3::Xaxis = IEVector3(1.0f, 0.0f, 0.0f);
const IEVector3 IEVector3::Yaxis = IEVector3(0.0f, 1.0f, 0.0f);
const IEVector3 IEVector3::Zaxis = IEVector3(0.0f, 0.0f, 1.0f);
const IEVector3 IEVector3::ZeroVector = IEVector3(0.0f, 0.0f, 0.0f);

IEVector3::IEVector3() 
	: x(0.0f)
	, y(0.0f)
	, z(0.0f)
{}

IEVector3::IEVector3(float xx, float yy, float zz) 
	: x(xx)
	, y(yy)
	, z(zz)
{}

IEVector3::IEVector3(const float v[]) 
	: x(v[0])
	, y(v[1])
	, z(v[2])
{}

IEVector3::IEVector3(const IEVector4& cp) 
	: x(cp.getX())
	, y(cp.getY())
	, z(cp.getZ())
{}

void IEVector3::operator+=(const IEVector3& vector)
{
	x += vector.x;
	y += vector.y;
	z += vector.z;
}

void IEVector3::operator-=(const IEVector3& vector)
{
	x -= vector.x;
	y -= vector.y;
	z -= vector.z;
}

void IEVector3::operator*=(const IEVector3& vector)
{
	x *= vector.x;
	y *= vector.y;
	z *= vector.z;
}

void IEVector3::operator*=(float t)
{
	x *= t;
	y *= t;
	z *= t;
}

void IEVector3::operator/=(const IEVector3& vector)
{
	x /= vector.x;
	y /= vector.y;
	z /= vector.z;
}

void IEVector3::operator/=(float t)
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	x *= tinv;
	y *= tinv;
	z *= tinv;
}

IEVector3 IEVector3::operator+(const IEVector3& vector) const
{
	return IEVector3(	x + vector.x,
						y + vector.y,
						z + vector.z);
}

IEVector3 IEVector3::operator-(const IEVector3& vector) const
{
	return IEVector3(	x - vector.x,
						y - vector.y,
						z - vector.z);
}

IEVector3 IEVector3::operator-() const
{
	return IEVector3(-x, -y, -z);
}

IEVector3 IEVector3::operator*(const IEVector3& vector) const
{
	return IEVector3(	x * vector.x,
						y * vector.y,
						z * vector.z);
}

IEVector3 IEVector3::operator*(float t) const
{
	return IEVector3(	x * t,
						y * t,
						z * t);
}

IEVector3 IEVector3::operator/(const IEVector3& vector) const
{
	assert(vector.x != 0.0f && vector.y != 0.0f && vector.z != 0.0f);
	return IEVector3(	x / vector.x,
						y / vector.y,
						z / vector.z);
}

IEVector3 IEVector3::operator/(float t) const
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	return IEVector3(	x * tinv,
						y *	tinv,
						z * tinv);
}

float IEVector3::DotProduct(const IEVector3& vector) const
{
	return x * vector.x + y * vector.y + z * vector.z;
}

IEVector3 IEVector3::CrossProduct(const IEVector3& vector) const
{
	return IEVector3(	y * vector.z - z * vector.y,
						z * vector.x - x * vector.z,
						x * vector.y - y * vector.x);
}

float IEVector3::Length() const
{
	return sqrtf(x * x + y * y + z * z);
}

float IEVector3::LengthSqr() const
{
	return x * x + y * y + z * z;
}

IEVector3 IEVector3::Normalize() const
{
	float length = Length();
	if(length != 0.0f) 
		length = 1.0f / length;

	return IEVector3(	x * length,
						y * length,
						z * length);
}

IEVector3& IEVector3::NormalizeSelf()
{
	float length = Length();
	if(length != 0.0f) 
		length = 1.0f / length;

	x *= length;
	y *= length;
	z *= length;
	return *this;
}

bool IEVector3::operator==(const IEVector3& vector) const
{
	return  std::equal(v, v + 3, vector.v);
}

bool IEVector3::operator!=(const IEVector3& vector) const
{
	return !std::equal(v, v + 3, vector.v);
}

// Left Scalar Operators
IEVector3 operator*(float scalar, const IEVector3& vector)
{
	return  vector * scalar;
}
