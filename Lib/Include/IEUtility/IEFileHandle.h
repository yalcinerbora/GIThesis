/**
Platform Independant file path holder
Holds the data as in platform path
Needs the file seperators as forward slash

TODO: Add more usablity (kinda dull atm)

Author(s):
Bora Yalciner
*/

#ifndef __IE_FILEHANDLE_H__
#define __IE_FILEHANDLE_H__

#include "IESystemDefinitions.h"
#include "IEMetaProgramming.h"

using IEFileHandle = IEConstSwitch<static_cast<std::size_t>(IE_CURRENT_PLATFORM), 
									void*,	// WINDOWS
									int,	// LINUX
									int,	// MACOS
									int,	// XBOX
									int,	// PS
									int,	// ANDROID
									int>;	// WINDOWS
#endif //__IE_FILEHANDLE_H__