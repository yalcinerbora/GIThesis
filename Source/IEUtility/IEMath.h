#pragma once
/**

*/

#include <cmath>
#include <cassert>

namespace IEMathConstants
{
	static constexpr double	PI = 3.1415926535897932384626433;
	static constexpr double PISqr = PI * PI;
	static constexpr double InvPI = 1.0 / PI;
	static constexpr double InvPISqr = 1.0 / (PI * PI);
	static constexpr double Sqrt2 = 1.4142135623730950488016887;
	static constexpr double Sqrt3 = 1.7320508075688772935274463;
	static constexpr double E = 2.7182818284590452353602874;
	static constexpr double InvE = 1.0f / E;

	static constexpr double DegToRadCoef = PI / 180.0;
	static constexpr double RadToDegCoef = 180.0 / PI;

}

namespace IEMathFunctions
{
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

	inline float GeomSeries(unsigned int n, float a)
	{
		// a^0 + a^1 + .... + a^n
		assert(a != 1.0f);
		return static_cast<float>((1.0f - ::std::pow(a, n + 1)) / (1.0f - a));
	}

	inline float GeomSubSeries(unsigned int n0, unsigned int n1, float a)
	{
		assert(n0 <= n1);
		return GeomSeries(n1, a) - GeomSeries(n0, a);
	}

	template<class T>
	inline T SumLinear(T n)
	{
		return static_cast<T>(n * (n + 1) / 2.0f);
	}
}
