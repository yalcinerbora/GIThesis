/**

*/

#ifndef __IE_FUNCTIONS_H__
#define __IE_FUNCTIONS_H__

namespace IEFunctions
{
	template <class T>
	float Clamp(T x, T max, T min)
	{
		return x < max ? max : (x > min ? min : x);
	}
};
#endif //__IE_FUNCTIONS_H__