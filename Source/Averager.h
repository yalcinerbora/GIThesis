/**

Simple Average class

*/

#ifndef __AVERAGER_H__
#define __AVERAGE_H__

#include <type_traits>

template<class T>
class Averager
{
	static_assert(std::is_arithmetic<T>::value, "Not Artihmetic Type");

	private:
		unsigned int	count;
		T				average;

	public:
		Averager() : count(0), average(0) {}
		void Avg(T t) 
		{
			float ratio = (count / static_cast<float>(count + 1));
			average = (average * ratio) + (t / static_cast<float>(count + 1));
		}

		T Flush()
		{
			T avg = average;
			average = 0;
			count = 0;
			return avg;
		}
};
#endif //__AVERAGE_H__