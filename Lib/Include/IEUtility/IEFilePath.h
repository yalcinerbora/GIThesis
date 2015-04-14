/**
	Platform Independant file path holder
	Holds the data as in platform path
	Needs the file seperators as forward slash

	TODO: Add more usablity (kinda dull atm)

Author(s):
	Bora Yalciner
*/

#ifndef __IE_FILEPATH_H__
#define __IE_FILEPATH_H__

#include "IESystemDefinitions.h"
#include "IEStackString.h"

#define IE_FILE_PATH_DEFAULT_SIZE 256

template<unsigned int N, IEPlatformType platform>
class IEFilePathC
{
	private:
		static const IEStackString<2>	seperator;
		IEStackString<N>				pathString;

		void							ConvertToPlatformPath();

	protected:

	public:
		//Constructors & Destructor
										IEFilePathC();
										IEFilePathC(const char cString[]);
										IEFilePathC(const char cString[], int length);
										IEFilePathC(const std::string& stlString);
		template<unsigned int S>		IEFilePathC(const IEStackString<S>& stackString);
										IEFilePathC(const IEFilePathC<N, platform>&) = default;

										// GCC ERROR HERE 4.8.2 (Extern Templates does not link this)
										// Making implicit declaration
										//~IEFilePathC() = default;

		// Cast Operator
		operator						const char*();

		// Low level access
		const char*						Data() const;

		IEFilePathC<N, platform>&		operator/(const char cString[]);
		IEFilePathC<N, platform>&		operator/(const std::string& stlString);
		template<unsigned int S>
		IEFilePathC<N, platform>&		operator/(const IEStackString<S>&);
		template<unsigned int S>
		IEFilePathC<N, platform>&		operator/(const IEFilePathC<S, platform>&);

		// Change Directory to one level above (or one level up)
		void							DeleteLastSelf();
		IEFilePathC<N, platform>		DeleteLast() const;

		// Acces Last Element with no seperators (Maybe a File Name depending on situation)
		template<unsigned int S>
		IEStackString<S>				GetLastS() const;
		std::string						GetLastStl() const;

		// Getting Data Out of this class to standard containers
		const IEStackString<N>&			getPathSString() const;
		std::string						getPathStlString() const;

		// Util
		bool							IsEmpty() const;
		int								Length() const;
		int								MaxLength() const;
		void							Clear();
};

// FilePath Alias
template<unsigned int N = IE_FILE_PATH_DEFAULT_SIZE>
using IEFilePath = IEFilePathC<N, IE_CURRENT_PLATFORM>;

#include "IEFilePath.hpp"

// Force Extern for some common types
extern template class IEFilePathC<64u, IE_CURRENT_PLATFORM>;
extern template class IEFilePathC<128u, IE_CURRENT_PLATFORM>;
extern template class IEFilePathC<256u, IE_CURRENT_PLATFORM>;
extern template class IEFilePathC<512u, IE_CURRENT_PLATFORM>; 

// extern template class IEFilePath<64>;
// extern template class IEFilePath<128>;
// extern template class IEFilePath<256>;
// extern template class IEFilePath<512>; 

#endif //__IE_FILEPATH_H__