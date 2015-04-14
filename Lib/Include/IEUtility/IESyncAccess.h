/**

Author(s):
Bora Yalciner
*/
#ifndef __IE_SYNCACCESS_H__
#define __IE_SYNCACCESS_H__

template <class T>
class IESyncAccess
{
	private:
	T			t1;
	T			t2;

	T*			waitT;
	T*			usedT;

	protected:

	public:
				IESyncAccess();
				IESyncAccess(const T&, const T&);
				~IESyncAccess();

	void		Swap();
	void		SetWait(const T&) const;
	T*			GetWait() const;
	T*			GetUsed() const;

};

template <class T>
IESyncAccess<T>::IESyncAccess()
{
	waitT = &t1;
	usedT = &t2;
}

template <class T>
IESyncAccess<T>::IESyncAccess(const T& t1, const T& t2) : t1(t1), t2(t2)
{
	waitT = &t1;
	usedT = &t2;
}

template <class T>
IESyncAccess<T>::~IESyncAccess()
{}

template <class T>
void IESyncAccess<T>::SetWait(const T& t) const
{
	*waitT = t;
}

template <class T>
void IESyncAccess<T>::Swap()
{
	std::swap(waitT, usedT);
}

template <class T>
T* IESyncAccess<T>::GetWait() const
{
	return waitT;
}

template <class T>
T* IESyncAccess<T>::GetUsed() const
{
	return usedT;
}

#endif //__IE_SYNCACCESS_H__
