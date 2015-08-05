/**

*/

#ifndef __IE_MATH_H__
#define __IE_MATH_H__

#include <cmath>

namespace IEMath
{
	// PI Stuff
	const float PI = 3.14159265f;
	const float PISqr = PI * PI;
	const float InvPI = 1.0f / PI;
	const float InvPISqr = 1.0f / (PI * PI);

	// E Stuff
	const float E = 3.14159265f;
	const float InvE = 1.0f / E;

	inline float CosF(float radians)
	{
		return cosf(radians);
	}

	inline float SinF(float radians)
	{
		return sinf(radians);
	}

	inline float TanF(float radians)
	{
		return tanf(radians);
	}

	inline float ACosF(float radians)
	{
		return acosf(radians);
	}

	inline float ASinF(float radians)
	{
		return asinf(radians);
	}

	inline float ATanF(float radians)
	{
		return atanf(radians);
	}

	inline float AbsF(float number)
	{
		return fabsf(number);
	}

	inline float ToRadians(float degree)
	{
		static const float DegToRadCoef = PI / 180.0f;
		return degree * DegToRadCoef;
	}

	inline float ToDegrees(float radian)
	{
		static const float RadToDegCoef = 180.0f / PI ;
		return radian * RadToDegCoef;
	}

	inline float LogF(float number)
	{
		return logf(number);
	}

	inline float Log10F(float number)
	{
		return log10f(number);
	}

	inline float Log2F(float number)
	{
		return log2f(number);
	}

	inline unsigned int UpperPowTwo(unsigned int number)
	{
		static_assert(sizeof(unsigned int) == 4, "UpperPowTwo only works on 32 bit integers");
		if(number <= 1) return 2;

		number--;
		number |= number >> 1;
		number |= number >> 2;
		number |= number >> 4;
		number |= number >> 8;
		number |= number >> 16;
		number++;

		return number;
	}

}
#endif //__IE_MATH_H__