/**
	
	File I/O Wrapper of C++. Little bit comple and has better error representation.
	Also supports  FliePath Class and holds no dynamic data. (All Stack)
	Also as RAII as possible

Author(s):
	Bora Yalciner

*/
#ifndef __IE_FILE_H__
#define __IE_FILE_H__

#include "IEFilePath.h"
#include "IEFileHandle.h"
#include "IETypes.h"

enum class IEFileErrorType
{
	// Errors
	OK,

	// Open Related
	ALREADY_OPENED_A_FILE,		// This File Class already holds an open file.
	FILE_NOT_EXISTS,
	FILE_EXISTS,
	CANT_READ_NEW_FILE,			// You cant read a file that is created

	// Generic
	PERMISSION_ERROR,
	PATH_SYNTAX_ERROR,
	READ_BUFFER_TOO_SMALL,
	FATAL_ERROR					
};

enum class IEFileActionType
{
	CREATE,			// Returns a new file only if there is no file specified by the path
	OVERWRITE,		// Overwrites the file only if file exists, old contents are deleted.
	CREATE_FORCE,	// Overwrites the file always, dont care files existance or not.
	ACCESS,			// Opens file, Returns not exist if file not exists
	ACCESS_CREATE,	// Opens the file if file exits or creates then opens the file.
};

enum class IEFileShareType
{
	CONSERVATIVE,	// Hold the file do whatever with it.
	SHARED_READ,	// Write only, ppl can access this file only for reading.
	SHARED_WRITE,	// Read only, ppl can access this file only for writing.
	SHARED_RW		// R-W, ppl can access this file for both reading and writing.
					// Noone can delete/rename the file on this modes
};

enum class IEFileAccessType
{
	READ,
	WRITE,
	BOTH
};

enum class IEFileAccessHintType
{
	SEQUENTIAL,
	RANDOM,
};

class IEFile
{
	private:
		IEFileHandle				handle;

		IEUInt64					relativePtrLocation;
		bool						fileOpened;
		IEUInt64					fileSize;
		bool						eofReached;

	protected:

	public:
		// Constructors & Destructors
									IEFile();
									IEFile(const IEFile&) = delete;
									~IEFile();

		// Accessors										
		bool						hasFileOpened() const;
		IEUInt64					getFileSize() const;
		bool						isEOFReached() const;
		IEUInt64					getRelativePtrLocation() const;

		// Platform Specific
		// File Access
		IEFileErrorType				Open(const char*,
										 IEFileActionType,
										 IEFileAccessType,
										 IEFileShareType,
										 IEFileAccessHintType = IEFileAccessHintType::SEQUENTIAL);
		IEFileErrorType				Close();

		// File Operations
		IEFileErrorType				Read(IEByte buffer[], IEUInt32& actualAmount, IEUInt32 requestedAmount);
		IEFileErrorType				ReadRest(IEByte buffer[], IESize bufferSize);
		IEFileErrorType				Write(IEUInt32& actualAmount, const IEByte buffer[], IEUInt32 requestedAmount);

		// Static Functions
		static IEFileErrorType		ReadAll(IEByte data[], IESize arrayLength,
											const char* path,
											IEFileShareType);
		static IEFileErrorType		Delete(const char* path);
		static IEFileErrorType		Move(const char* newFile, const char* existingFile);
		static IEFileErrorType		Copy(const char* newFile, const char* existingFile);
		static IEFileErrorType		FileExists(const char* path);
};
#endif //__IE_FILE_H__
