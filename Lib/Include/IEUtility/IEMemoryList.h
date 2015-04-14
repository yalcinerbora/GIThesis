/*
Thread Safe
Data Structure (Linked List)

Pointers to the Elements in the List
Stays Valid

Used In Renderer's Memory Managment Class
MemoryManger Deletes the Incoming MemoryElements(In its own thread)
Other Threads Create Objects In the List (Arbitrary Threads which cosntructs the scene)
Actual Creation of the Object(In the GPU memory) also done by Memory Manager Thead

TODO::
-- Optimize Access (Multiple Reads Should Happen whine nothing gets written) 
					(immutable funcs should be called async if muttable func is not avail)

Author(s):
Bora Yalciner
*/

#ifndef __IE_MEMORYLIST_H__
#define __IE_MEMORYLIST_H__

#include <list>
#include "IEThread.h"

template <class T>
class IEMemoryList
{
	private:

	protected:
		std::list<T>			dataList;
		mutable IEMutex			mutex;
		
	public:
		// Constructors & Destructors
								IEMemoryList();
								IEMemoryList(const IEMemoryList<T>&) = delete;
		IEMemoryList<T>&		operator=(const IEMemoryList<T>&) = delete;
								~IEMemoryList() = default;

		// Interface
		template <class... Args>
		T&					 	EmplaceAppend(Args...);
		bool					Delete(T&, bool(*CustomEqualFunc)(const T&, const T&));
		bool					Delete(T&);
		void					DeleteAll();

		template <class R, class... Args>
		void					CallFunctionAll(R(T::*func) (Args...),
												Args... arguments);

		template <class R, class... Args>
		void					CallFunctionAll(R(T::*func) (Args...) const,
												Args... arguments) const;
};
//To Put Implementations on other file for template classes
#include "IEMemoryList.hpp"
#endif //__IE_MEMORYLIST_H__