#ifndef USE_AVX
#include <algorithm>
#include <cassert>

#include "IEVector4.h"
#include "IEVector3.h"
#include "IEFunctions.h"

const IEVector4 IEVector4::ZeroVector = IEVector4(0.0f, 0.0f, 0.0f, 0.0f);
const IEVector4 IEVector4::XAxis = IEVector4(1.0f, 0.0f, 0.0f, 0.0f);
const IEVector4 IEVector4::YAxis = IEVector4(0.0f, 1.0f, 0.0f, 0.0f);
const IEVector4 IEVector4::ZAxis = IEVector4(0.0f, 0.0f, 1.0f, 0.0f);

IEVector4::IEVector4(const IEVector3& cp) 
	: x(cp.getX())
	, y(cp.getY())
	, z(cp.getZ())
	, w(1.0f)
{}

IEVector4::IEVector4(const IEVector3& cp, float w) 
	: x(cp.getX())
	, y(cp.getY())
	, z(cp.getZ())
	, w(w)
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
	return std::sqrt(x * x + y * y + z * z + w * w);
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

IEVector4 IEVector4::Clamp(const IEVector4& min, const IEVector4& max) const
{
	return
	{
		(x < min.x) ? min.x : ((x > max.x) ? max.x : x),
		(y < min.y) ? min.y : ((y > max.y) ? max.y : y),
		(z < min.z) ? min.z : ((z > max.z) ? max.z : z),
		(w < min.w) ? min.w : ((w > max.w) ? max.w : w)
	};
}

IEVector4 IEVector4::Clamp(float min, float max) const
{
	return
	{
		(x < min) ? min : ((x > max) ? max : x),
		(y < min) ? min : ((y > max) ? max : y),
		(z < min) ? min : ((z > max) ? max : z),
		(w < min) ? min : ((w > max) ? max : w)
	};
}

IEVector4& IEVector4::ClampSelf(const IEVector4&  min, const IEVector4& max)
{
	x = (x < min.x) ? min.x : ((x > max.x) ? max.x : x);
	y = (y < min.y) ? min.y : ((y > max.y) ? max.y : y);
	z = (z < min.z) ? min.z : ((z > max.z) ? max.z : z);
	w = (w < min.w) ? min.w : ((w > max.w) ? max.w : w);
	return *this;
}

IEVector4& IEVector4::ClampSelf(float min, float max)
{
	x = (x < min) ? min : ((x > max) ? max : x);
	y = (y < min) ? min : ((y > max) ? max : y);
	z = (z < min) ? min : ((z > max) ? max : z);
	return *this;
}

bool IEVector4::operator==(const IEVector4& vector) const
{
	return std::equal(v, v + VectorW, vector.v);
}

bool IEVector4::operator!=(const IEVector4& vector) const	
{
	return !std::equal(v, v + VectorW, vector.v);
}

// Left Scalar operators
IEVector4 operator*(float scalar, const IEVector4& vector)
{
	return vector * scalar;
}

template<>
IEVector4 IEFunctions::Lerp(const IEVector4& start, const IEVector4& end, float percent)
{
	percent = IEFunctions::Clamp(percent, 0.0f, 1.0f);
	return (start + percent * (end - start));
}

template<>
IEVector4 IEFunctions::Clamp(const IEVector4& vec, const IEVector4& min, const IEVector4& max)
{
	return vec.Clamp(min, max);
}
#endif // USE_AVX