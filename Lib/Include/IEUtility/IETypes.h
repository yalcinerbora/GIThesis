/**


Author(s):
	Bora Yalciner
*/

#ifndef __IE_TYPES_H__
#define __IE_TYPES_H__

#include <stdint.h>
#include <cstdlib>
#include <limits>

// IE Byte is special, it represents basic machine units 
// it "should" be 8bits (minimal ISA workable data type)
typedef uint8_t		IEByte;

typedef int8_t		IEInt8;
typedef uint8_t		IEUInt8;

typedef int16_t		IEInt16;
typedef uint16_t	IEUInt16;

typedef int32_t		IEInt32;
typedef uint32_t	IEUInt32;

typedef int64_t		IEInt64;
typedef uint64_t	IEUInt64;

typedef size_t		IESize;

// Some Sanity Checks Here
static_assert(sizeof(IEByte) == 1, "Byte Size Sanity Check Fail.");

static_assert(sizeof(IEInt8) == 1, "Int8 Size Sanity Check Fail.");
static_assert(sizeof(IEUInt8) == 1, "UInt8 Size Sanity Check Fail.");

static_assert(sizeof(IEInt16) == 2, "Int16 Size Sanity Check Fail.");
static_assert(sizeof(IEUInt16) == 2, "UInt16 Size Sanity Check Fail.");

static_assert(sizeof(IEInt32) == 4, "Int32 Size Sanity Check Fail.");
static_assert(sizeof(IEUInt32) == 4, "UInt32 Size Sanity Check Fail.");

static_assert(sizeof(IEInt64) == 8, "Int64 Size Sanity Check Fail.");
static_assert(sizeof(IEUInt64) == 8, "UInt64 Size Sanity Check Fail.");

static_assert(sizeof(IESize) == 8, "IESize Size Sanity Check Fail.");

static_assert(sizeof(float) == 4, "Float Size Sanity Check Fail.");
static_assert(sizeof(double) == 8, "Double Size Sanity Check Fail.");

// Assuming theese are 2's complement (prob is)
static const IEByte		IE_MAX_BYTE = std::numeric_limits<IEByte>::max();
static const IEByte		IE_MIN_BYTE = std::numeric_limits<IEByte>::min();

static const IEUInt8	IE_MAX_UINT8 = std::numeric_limits<IEUInt8>::max();
static const IEUInt8	IE_MIN_UINT8 = std::numeric_limits<IEUInt8>::min();

static const IEInt8		IE_MAX_INT8 = std::numeric_limits<IEInt8>::max();
static const IEInt8		IE_MIN_INT8 = std::numeric_limits<IEInt8>::min();

static const IEUInt16	IE_MAX_UINT16 = std::numeric_limits<IEUInt16>::max();
static const IEUInt16	IE_MIN_UINT16 = std::numeric_limits<IEUInt16>::min();

static const IEInt16	IE_MAX_INT16 = std::numeric_limits<IEInt16>::max();
static const IEInt16	IE_MIN_INT16 = std::numeric_limits<IEInt16>::min();

static const IEUInt32	IE_MAX_UINT32 = std::numeric_limits<IEUInt32>::max();
static const IEUInt32	IE_MIN_UINT32 = std::numeric_limits<IEUInt32>::min();

static const IEInt32	IE_MAX_INT32 = std::numeric_limits<IEInt32>::max();
static const IEInt32	IE_MIN_INT32 = std::numeric_limits<IEInt32>::min();

static const IEUInt64	IE_MAX_UINT64 = std::numeric_limits<IEUInt64>::max();
static const IEUInt64	IE_MIN_UINT64 = std::numeric_limits<IEUInt64>::min();

static const IEInt64	IE_MAX_INT64 = std::numeric_limits<IEInt64>::max();
static const IEInt64	IE_MIN_INT64 = std::numeric_limits<IEInt64>::min();

static const IESize		IE_MAX_SIZE = std::numeric_limits<std::size_t>::max();
static const IESize		IE_MIN_SIZE = std::numeric_limits<std::size_t>::min();

#endif//__IE_TYPES_H__