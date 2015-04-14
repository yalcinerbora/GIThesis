/**

Object Oriented Primitives
NonCopyable class etc..

Author(s):
	Bora Yalciner
*/
#ifndef __IE_OOPRIMITIVES_H__
#define __IE_OOPRIMITIVES_H__

// Interface that makes your class noncopyable
class NoCopyI
{
	protected:
						NoCopyI() = default;

	public:
						NoCopyI(const NoCopyI&) = delete;
		NoCopyI&		operator=(const	NoCopyI&) = delete;
};
#endif //__IE_OOPRIMITIVES_H__