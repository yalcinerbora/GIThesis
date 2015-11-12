/**

Macros 

Preprocessor Stuff for ease of using

*/
#ifndef __MACROS_H__
#define __MACROS_H__

#include <iostream>

// Debug
#ifdef GI_DEBUG
	#define GI_DEBUG_LOG(string, ...) printf(string"\n", ## __VA_ARGS__ )
#else
	#define GI_DEBUG_LOG(...)
#endif

// Errors
#define GI_ERROR_LOG(string, ...) fprintf( stderr, string"\n", ## __VA_ARGS__ )

// Log
#define GI_LOG(string, ...) fprintf( stdout, string"\n", ## __VA_ARGS__ )

#include <cassert>

#endif //__MACROS_H__