/**

*/

#ifndef __IE_FUNCTIONS_H__
#define __IE_FUNCTIONS_H__

namespace IEFunctions
{
	template <class T>
	float Clamp(T x, T min, T max)
	{
		return (x < min) ? min : ((x > max) ? max : x);
	}

	template <class T>
	T Lerp(const T& from, const T& to, float alpha)
	{
		assert(alpha >= 0.0f && alpha <= 1.0f);
		return to * alpha + from * (1.0f - alpha);
	}
};
#endif //__IE_FUNCTIONS_H__