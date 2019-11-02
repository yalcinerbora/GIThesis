#include <algorithm>
#include <cassert>

#include "IEVector2.h"
#include "IEFunctions.h"
#include <cmath>

// Constants
const IEVector2 IEVector2::Xaxis = IEVector2(1.0f, 0.0f);
const IEVector2 IEVector2::Yaxis = IEVector2(0.0f, 1.0f);
const IEVector2 IEVector2::ZeroVector = IEVector2(0.0f, 0.0f);

void IEVector2::operator+=(const IEVector2& vector)
{
	x += vector.x;
	y += vector.y;
}

void IEVector2::operator-=(const IEVector2& vector)
{
	x -= vector.x;
	y -= vector.y;
}

void IEVector2::operator*=(const IEVector2& vector)
{
	x *= vector.x;
	y *= vector.y;
}

void IEVector2::operator*=(float t)
{
	x *= t;
	y *= t;
}

void IEVector2::operator/=(const IEVector2& vector)
{
	x /= vector.x;
	y /= vector.y;
}

void IEVector2::operator/=(float t)
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	x *= tinv;
	y *= tinv;
}

IEVector2 IEVector2::operator+(const IEVector2& vector) const
{
	return IEVector2(x + vector.x,
					 y + vector.y);
}

IEVector2 IEVector2::operator-(const IEVector2& vector) const
{
	return IEVector2(x - vector.x,
					 y - vector.y);
}

IEVector2 IEVector2::operator-() const
{
	return IEVector2(-x, -y);
}

IEVector2 IEVector2::operator*(const IEVector2& vector) const
{
	return IEVector2(x * vector.x,
					 y * vector.y);
}

IEVector2 IEVector2::operator*(float t) const
{
	return IEVector2(x * t,
					 y * t);
}

IEVector2 IEVector2::operator/(const IEVector2& vector) const
{
	assert(vector.x != 0.0f && vector.y != 0.0f);
	return IEVector2(x / vector.x,
					 y / vector.y);
}

IEVector2 IEVector2::operator/(float t) const
{
	assert(t != 0.0f);
	float tinv = 1.0f / t;
	return IEVector2(x * tinv,
					 y * tinv);
}

float IEVector2::DotProduct(const IEVector2& vector) const
{
	return x * vector.x + y * vector.y;
}

float IEVector2::Length() const
{
	return std::sqrt(x * x + y * y);
}

float IEVector2::LengthSqr() const
{
	return x * x + y * y;
}

IEVector2 IEVector2::Normalize() const
{
	float length = Length();
	if(length != 0.0f)
		length = 1.0f / length;

	return IEVector2(x * length,
					 y * length);
}

IEVector2& IEVector2::NormalizeSelf()
{
	float length = Length();
	if(length != 0.0f)
		length = 1.0f / length;

	x *= length;
	y *= length;
	return *this;
}

IEVector2 IEVector2::Clamp(const IEVector2& min, const IEVector2& max) const
{
	return
	{
		(x < min.x) ? min.x : ((x > max.x) ? max.x : x),
		(y < min.y) ? min.y : ((y > max.y) ? max.y : y)
	};
}

IEVector2 IEVector2::Clamp(float min, float max) const
{
	return
	{
		(x < min) ? min : ((x > max) ? max : x),
		(y < min) ? min : ((y > max) ? max : y)
	};
}

IEVector2& IEVector2::ClampSelf(const IEVector2&  min, const IEVector2& max)
{
	x = (x < min.x) ? min.x : ((x > max.x) ? max.x : x);
	y = (y < min.y) ? min.y : ((y > max.y) ? max.y : y);
	return *this;
}

IEVector2& IEVector2::ClampSelf(float min, float max)
{
	x = (x < min) ? min : ((x > max) ? max : x);
	y = (y < min) ? min : ((y > max) ? max : y);
	return *this;
}

bool IEVector2::operator==(const IEVector2& vector) const
{
	return  std::equal(v, v + VectorW, vector.v);
}

bool IEVector2::operator!=(const IEVector2& vector) const
{
	return !std::equal(v, v + VectorW, vector.v);
}

// Left Scalar Operators
IEVector2 operator*(float scalar, const IEVector2& vector)
{
	return  vector * scalar;
}

template<>
IEVector2 IEFunctions::Lerp(const IEVector2& start, const IEVector2& end, float percent)
{
	percent = IEFunctions::Clamp(percent, 0.0f, 1.0f);
	return (start + percent * (end - start));
}

template<>
IEVector2 IEFunctions::Clamp(const IEVector2& vec, const IEVector2& min, const IEVector2& max)
{
	return vec.Clamp(min, max);
}