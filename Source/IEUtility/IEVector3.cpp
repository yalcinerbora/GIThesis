#ifndef USE_AVX
#include <algorithm>
#include <cassert>

#include "IEVector3.h"
#include "IEVector4.h"
#include "IEFunctions.h"
#include <cmath>

// Constants
const IEVector3 IEVector3::XAxis = IEVector3(1.0f, 0.0f, 0.0f);
const IEVector3 IEVector3::YAxis = IEVector3(0.0f, 1.0f, 0.0f);
const IEVector3 IEVector3::ZAxis = IEVector3(0.0f, 0.0f, 1.0f);
const IEVector3 IEVector3::ZeroVector = IEVector3(0.0f, 0.0f, 0.0f);

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
	return std::sqrt(LengthSqr());
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

IEVector3 IEVector3::Clamp(const IEVector3& min, const IEVector3& max) const
{
	return
	{
		(x < min.x) ? min.x : ((x > max.x) ? max.x : x),
		(y < min.y) ? min.y : ((y > max.y) ? max.y : y),
		(z < min.z) ? min.z : ((z > max.z) ? max.z : z)
	};
}

IEVector3 IEVector3::Clamp(float min, float max) const
{
	return
	{
		(x < min) ? min : ((x > max) ? max : x),
		(y < min) ? min : ((y > max) ? max : y),
		(z < min) ? min : ((z > max) ? max : z)
	};
}

IEVector3& IEVector3::ClampSelf(const IEVector3&  min, const IEVector3& max)
{
	x = (x < min.x) ? min.x : ((x > max.x) ? max.x : x);
	y = (y < min.y) ? min.y : ((y > max.y) ? max.y : y);
	z = (z < min.z) ? min.z : ((z > max.z) ? max.z : z);
	return *this;
}

IEVector3& IEVector3::ClampSelf(float min, float max)
{
	x = (x < min) ? min : ((x > max) ? max : x);
	y = (y < min) ? min : ((y > max) ? max : y);
	z = (z < min) ? min : ((z > max) ? max : z);
	return *this;
}

bool IEVector3::operator==(const IEVector3& vector) const
{
	return  std::equal(v, v + VectorW, vector.v);
}

bool IEVector3::operator!=(const IEVector3& vector) const
{
	return !std::equal(v, v + VectorW, vector.v);
}

// Left Scalar Operators
IEVector3 operator*(float scalar, const IEVector3& vector)
{
	return  vector * scalar;
}

template<>
IEVector3 IEFunctions::Lerp(const IEVector3& start, const IEVector3& end, float percent)
{
	percent = IEFunctions::Clamp(percent, 0.0f, 1.0f);
	return (start + percent * (end - start));
}

template<>
IEVector3 IEFunctions::Clamp(const IEVector3& vec, const IEVector3& min, const IEVector3& max)
{
	return vec.Clamp(min, max);
}
#endif // USE_AVX