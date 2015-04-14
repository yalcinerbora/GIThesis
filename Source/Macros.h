/**

Macros 

Preprocessor Stuff for ease of using

*/
#ifndef __MACROS_H__
#define __MACROS_H__

// Debug
#ifdef GI_DEBUG
	static const bool DEBUG = true;
	#define GI_DEBUG_LOG(string, ...) printf(string"\n", ## __VA_ARGS__ )
#else
	static const bool DEBUG = false;
	#define GI_DEBUG_LOG(...)
#endif

// Errors
#define GI_ERROR_LOG(string, ...) fprintf( stderr, string"\n", ## __VA_ARGS__ )

// Log
#define GI_LOG(string, ...) fprintf( stdout, string"\n", ## __VA_ARGS__ )

#include <cassert>

#endif //__MACROS_H__