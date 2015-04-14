/**
	Chain of Null Terminated Pointers

Author(s):
	Bora Yalciner
*/
#ifndef __IE_CHAIN_H__
#define __IE_CHAIN_H__

#include "IETypes.h"
#include <algorithm>

template <class T, IESize MaxAmount>
struct IEChain
{
		T*			data[MaxAmount + 1];
					IEChain() {std::fill_n(data, data + MaxAmount + 1, nullptr);}
};
#endif //__IE_CHAIN_H__