/*
Dynamic Array (std::deque wrap)
With Concurrent Access
Can Call Class Functions Safely (Thread Safe)


TODO::
	-- Optimize Access (Multiple Reads Should Happen whine nothing gets written) 
						(immutable funcs should be called async if muttable func is not avail)

Author(s):
Bora Yalciner
*/

#ifndef __IE_CLASSLIST_H__
#define __IE_CLASSLIST_H__

#include "IEList.h"

template <class T>
class IEClassList : public IEList<T>
{
	private:

	protected:

	public:
		// Constructors & Destructor
									IEClassList();
									IEClassList(const IEList<T>&);
									~IEClassList();

		// Function Calling
		template <class R, class... Args>
		R							CallFunction(int position,
													R (T::*func) (Args...),
													Args... arguments);

		template <class R, class... Args>
		R							CallFunction(int position,
													R(T::*func) (Args...) const,
													Args... arguments) const;

		template <class R, class... Args>
		void						CallFunctionAll(R(T::*func) (Args...),
													Args... arguments);

		template <class R, class... Args>
		void						CallFunctionAll(R(T::*func) (Args...) const,
													Args... arguments) const;


		template <class Functor>
		void						ApplyFunction(int position,
													Functor&);

		template <class Functor>
		void						ApplyFunctionAll(Functor&);
};

template <class T>
class IEClassList<T*> : public IEList<T*>
{
	private:

	protected:

	public:
		// Constructors & Destructor
									IEClassList();
									IEClassList(const IEList<T*>&);
									~IEClassList();

		// Function Calling
		template <class R, class... Args>
		R							CallFunction(int position,
													R(T::*func) (Args...),
													Args... arguments);

		template <class R, class... Args>
		R							CallFunction(int position,
													R(T::*func) (Args...) const,
													Args... arguments) const;

		template <class R, class... Args>
		void						CallFunctionAll(R(T::*func) (Args...),
													Args... arguments);

		template <class R, class... Args>
		void						CallFunctionAll(R(T::*func) (Args...) const,
													Args... arguments) const;

		template <class Functor>
		void						ApplyFunction(int position,
													Functor&);
		
		template <class Functor>
		void						ApplyFunctionAll(Functor&);
};
#include "IEClassList.hpp"
#endif //__IE_CLASSLIST_H__