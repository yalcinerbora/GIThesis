/**

Author(s):
	Bora Yalciner

*/

#ifndef __IE_IESYSTEMDEFINITIONS_H__
#define __IE_IESYSTEMDEFINITIONS_H__

enum class IEPlatformType
{
	WINDOWS,
	LINUX,			// Hopefully
	MACOS,			// Hopefully
	XBOX,			// Hopefully
	PLAYSTATION,	// Hopefully
	ANDROID,		// Hopefully
	IOS,			// Hopefully
};

enum class IERendererType
{
	OPENGL_44,
	VULKAN,			// Hopefully
	OPENGL_ES_30,	// Hopefully
	DIRECTX_11,		// Hopefully
	DIRECTX_12,		// Hopefully
};

// Compiler Switch For the Current System
// Current System Switch
// Single Handle to specifiy platform
#if defined _WIN32
	#define IE_WINDOWS_PLATFORM
	static const IEPlatformType IE_CURRENT_PLATFORM = IEPlatformType::WINDOWS;
#elif defined __APPLE__ && __MACH__
	#define IE_MACOS_PLATFORM
	static const IEPlatformType IE_CURRENT_PLATFORM = IEPlatformType::MACOS;
#elif defined __gnu_linux__
	#define IE_LINUX_PLATFORM
	static const IEPlatformType IE_CURRENT_PLATFORM = IEPlatformType::LINUX;
#else
	#error Fatal Error! Unspecified Platform.
#endif

#endif //__IE_IESYSTEMDEFINITIONS_H__