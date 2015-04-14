/**

	MetaProgramming Stuff
	Templates and such

Author(s):
	Bora Yalciner
*/
#ifndef __IE_METAPROGRAMMING_H__
#define __IE_METAPROGRAMMING_H__

#include <tuple>

// Usage
// IEConstSwitch<N, type, type, .... >
// it will choose the Nth type (0 indexed) type as a typename
// IEConstSwitch<0, long, int> name;  -> long name;
// IEConstSwitch<1, long, int> name;  -> int name;
// Its on compile time
template<std::size_t N, typename... T>
using IEConstSwitch = typename std::tuple_element<N, std::tuple<T...> >::type;

// Static Polymorphism Helper Macros
#define IE_STATIC_POLY_FUNC_IMPL(function, funcName, ...) \
	function \
	{ \
		static_cast<Implementation*>(this)->funcName ## Impl(## __VA_ARGS__); \
	} \

// Crappy Inheritence(Merge) of two Enums
template<typename Base, typename Extend>
struct IEEnumMerge
{
	bool isBase;
	union
	{
		Base baseEnum;
		Extend extendEnum;
	};

	// Implict Casting
	operator Base() {return baseEnum;}
	operator Extend() {return extendEnum;}

	// Constructors
	IEEnumMerge(Base e) : isBase(true), baseEnum(e) {}
	IEEnumMerge(Extend e) : isBase(false), extendEnum(e) {}

	// Assignment Operators
	bool operator==(Base e) const {return isBase ? (e == baseEnum) : false;}
	bool operator==(Extend e) const {return isBase ? false : e == extendEnum;};
	bool operator!=(Base e) const {return !(*this == e);}
	bool operator!=(Extend e) const {return !(*this == e);};
};

template<typename Base, typename Extend>
static bool operator==(Base base, IEEnumMerge<Base, Extend> e) {return e == base;}

template<typename Base, typename Extend>
static bool operator==(Extend extend, IEEnumMerge<Base, Extend> e) {return e == extend;};

template<typename Base, typename Extend>
static bool operator!=(Base base, IEEnumMerge<Base, Extend> e) {return e != base;};

template<typename Base, typename Extend>
static bool operator!=(Extend extend, IEEnumMerge<Base, Extend> e) {return e != extend;};

#endif //__IE_METAPROGRAMMING_H__