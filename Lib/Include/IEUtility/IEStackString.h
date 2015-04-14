/**
	string static array
	usefull for small strings to reduce fragmantation of memory
	very basic static c string array in a class

Author(s):
	Bora Yalciner
*/

#ifndef __IE_STACKSTRING_H__
#define __IE_STACKSTRING_H__

#define IE_STACK_STRING_DEFAULT_LENGTH	32
#define IE_NOT_FOUND					-1

#include <string>

template <unsigned int N = IE_STACK_STRING_DEFAULT_LENGTH>
class IEStackString
{
	private:
		char							dataArray[N];
		int								length;

	protected:

	public:
		// Constructors & Destructor
										IEStackString();
										IEStackString(const char cString[]);
										IEStackString(const char cString[], int length);
										IEStackString(const std::string& stlString);
		template <unsigned int S>		IEStackString(const IEStackString<S>&);
										~IEStackString();

		// Accessors
		char							CharAt(int) const;
		const char*						Data() const;
	
		// Mutators
		void							Replace(int position, char data);
		void							Replace(int position, const char data[], int length);
		char*							Edit();
		void							LengthAfterEdit(int);
		void							LineEndingAfterEdit();

		// Cast Operator
		operator						const char*();

		template <unsigned int S>
		const IEStackString<N>&			operator=(const IEStackString<S>&);
		const IEStackString<N>&			operator=(const std::string& stlString);
		const IEStackString<N>&			operator=(const char cString[]);

		// Logic
		template <unsigned int S>
		bool							operator==(const IEStackString<S>&) const;
		template <unsigned int S>
		bool							operator!=(const IEStackString<S>&) const;

		// Manipulation
		void							Append(char);
		template <unsigned int S>
		void							Append(const IEStackString<S>&);
		void							Append(const std::string&);
		void							Append(const char cString[]);
		void							Append(const char cString[], int size);
		IEStackString<N>				SubString(int start, int end) const;	// [start, end)
		IEStackString<N>&				Trim(int start, int end);				// [start, end)
		template <unsigned int R, unsigned int S>
		IEStackString<R>				Concatenate(const IEStackString<S>&) const;
			
		// Index Finding
		int								FirstIndexOf(char) const;
		int								FirstIndexOf(const char[], int size) const;
		template <unsigned int S>
		int								FirstIndexOf(const IEStackString<S>&) const;
		int								LastIndexOf(char) const;
		int								LastIndexOf(const char[], int size) const;
		template <unsigned int S>
		int								LastIndexOf(const IEStackString<S>&) const;

		// Util
		bool							IsEmpty() const;
		int								Length() const;
		int								MaxLength() const;
		void							Clear();

		// Misc
		std::string						ToStlString() const;
};
// To Put Implementations on other file for template classes
#include "IEStackString.hpp"

// Force Extern for some common types
extern template class IEStackString<>;
extern template class IEStackString<64>;
extern template class IEStackString<128>;
extern template class IEStackString<256>;

#endif //__IE_STACKSTRING_H__