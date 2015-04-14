/*
Non Thread Safe
Data Structure

Pointers to the Elements in the List
Stays Valid

Used In Abstract Factory Implementations to hold implemented classes
Has Minimal Amount of functions

Add and delete is constant time
Pointer and references of the elements stays valid if a new object is being added/deleted

Designed to hold relatively small objets
Pointer and element is coupled (to provide constant time deletes)

Author(s):
	Bora Yalciner
*/

#ifndef __IE_STATIONARYLIST_H__
#define __IE_STATIONARYLIST_H__

#include <allocators>
#include <type_traits>
#include "IETypes.h"
#include "IEOOPrimitives.h"
#include "IEMacros.h"

// Node Implementation
template <class T>
struct Node
{
	Node*	next;
	Node*	prev;
	T		element;

	template<class... Args>
	Node(Node* n, Node* p, Args... arg) : next(n), prev(p), element(arg...) {}
};

extern IESize IE_STATIONARY_LIST_CACHE_LINE_SIZE;

template <class T>
class IEStationaryList : public NoCopyI
{
	private:
		// Static Variable
		static std::allocator<Node<T>>	NODE_ALLOCATOR;
		static std::allocator<Node<T>*>	NODE_ARRAY_ALLOCATOR;
		static IESize					CHUNK_ELEMENT_COUNT;
		static const IEUInt32			CACHE_LINE_AMOUNT;
		static const IEUInt32			INITIAL_NODE_SIZE_ARRAY;
		static const float				SIZE_INCREASE_POLICY;

		// Properties
		Node<T>**				allocatedChunks;			// Allocated Chunks Array
		Node<T>*				head;					
		Node<T>*				deleteList;				
		IEUInt32				usedChunkCount;
		IEUInt32				totalChunkCount;

		// Utility Functions
		void					IncreaseChunkArray();
		void					AddChunk();
		Node<T>*				FindAllocLoc(T*) const;
		void					DeleteAlloc(Node<T>*);

	protected:
	public:
		// Constructors & Destructor
								IEStationaryList();
								IEStationaryList(IESize preAllocCount);
								~IEStationaryList();

		template <class... Args>
		T&					 	EmplaceAppend(Args...);
		void					Delete(T&);

};
#include "IEStationaryList.hpp"
// Force Extern Basic Types
IE_FORCE_EXTERN_BASIC_TYPES(IEStationaryList)
#endif //__IE_STATIONARYLIST_H__