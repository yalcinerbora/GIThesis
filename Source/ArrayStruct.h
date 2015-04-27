/**

Super Simple Array Structs

*/


#ifndef __ARRAYSTRUCT_H__
#define __ARRAYSTRUCT_H__

#include <cstdint>

template <class T>
struct Array32
{
	T*				arr;
	const uint32_t	length;
};

template <class T>
struct Array64
{
	T*			arr;
	uint64_t	length;
};
#endif //__ARRAYSTRUCT_H__