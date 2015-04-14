/*
Dynamic Array (std::deque wrap)
With Concurrent Access

All Operations Are Synchronized.
However, this may not suit your needs;
then you should implement from std::deque.

You can not do series of function calls in synchronized manner.


TODO::
	-- Optimize Access (Multiple Reads Should Happen whine nothing gets written) 
						(immutable funcs should be called async if muttable func is not avail)

Author(s):
	Bora Yalciner
*/

#ifndef __IE_LIST_H__
#define __IE_LIST_H__

#include <deque>
#include "IEThread.h"
#include "IEMacros.h"
#include "IETypes.h"

enum class IESortType
{
	IE_SORT_TYPE_QUICKSORT,
	IE_SORT_TYPE_INSERTIONSORT
};

template <class T>
class IEList
{
	private:
		
	protected:
		std::deque<T>		dataVector;
		mutable IEMutex		mutex;

		// Sort Functions
		void				ISort(bool(*ComparisonFunc) (const T&, const T&));

	public:
		// Constructors & Destructor
							IEList();
							IEList(const IEList<T>&);

							// GCC ERROR HERE 4.8.2 (Extern Templates does not link this)
							// Making implicit declaration
							//~IEList() = default;

		// Setting Memory
		void				Shrink();

		// Add, Delete & Read
		void				Append(const T&);
		void				Append(const IEList&);
		void				Add(int position, const T&);
		void				Add(int position, const IEList&);
		void				Add(int position, const T[], int size);
		void				Delete(int position);
		void				Delete(int position, int endPosition);
		bool				DeleteFirst(const T&);
		bool				DeleteFirst(const T&, bool (*CustomEqualFunc)(const T&, const T&));
		T					Fetch(int position);
		
		// Read & Find
		T					Read(int position) const;
		int					FindFirst(const T&) const;
		int					FindFirst(const T&, bool (*CustomEqualFunc)(const T&, const T&)) const;

		// Resetting Array
		void				SoftReset();	// Resets the Array without any Memory Operations
		void				HardReset();	// Resets the Array with Shrinking the Memory

		// Sort
		void				Sort(bool(*CustomCompareFunc)(const T&, const T&), IESortType = IESortType::IE_SORT_TYPE_QUICKSORT);

		// Operators
		T					operator[](int position) const;
		void				operator=(const IEList<T>&);

		// Util
		int					Size() const;
		bool				isEmpty() const;
};
//To Put Implementations on other file for template classes
#include "IEList.hpp"
// Force Extern Basic Types
IE_FORCE_EXTERN_BASIC_TYPES(IEList)
#endif //__IE_LIST_H__