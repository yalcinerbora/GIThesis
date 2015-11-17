/**

GFGDataType Enumeration
GFGIndexDataType Enumretaion
GFGVertexComponentLogic Enumeration
GFGDataTypeByteSize Array
GFGTopology Enumeration
GFGDirection Enumeration
GFGStringType Enumeration
GFGMaterialLogic Enumeration

Various enumerations used by the GFGHeader.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_ENUMERATIONS_H__
#define __GFG_ENUMERATIONS_H__

#include <cstdint>
#include <ostream>

enum class GFGDataType : uint32_t
{
	// Float Data Types
	HALF_1,				// one component IEEE 754 (16-bit) floating point number
	HALF_2,				// two component                   ""
	HALF_3,				// three component                 ""
	HALF_4,				// four component			       ""

	FLOAT_1,			// one component IEEE 754 (32-bit) floating point number
	FLOAT_2,			// two component                   ""
	FLOAT_3,			// three component                 ""
	FLOAT_4,			// four component			       ""

	DOUBLE_1,			// one component IEEE 754 (64-bit) floating point number
	DOUBLE_2,			// two component			       ""
	DOUBLE_3,			// three component			       ""
	DOUBLE_4,			// four component			       ""

	// Nobody will ever use this in this decade but w/e
	QUADRUPLE_1,		// one component IEEE 754 (128-bit) floating point number
	QUADRUPLE_2,		// two component			       ""
	QUADRUPLE_3,		// three component			       ""
	QUADRUPLE_4,		// four component			       ""

	// Integers
	// 8-bit
	INT8_1,
	INT8_2,
	INT8_3,
	INT8_4,

	UINT8_1,
	UINT8_2,
	UINT8_3,
	UINT8_4,

	// 16-bit
	INT16_1,
	INT16_2,
	INT16_3,
	INT16_4,

	UINT16_1,
	UINT16_2,
	UINT16_3,
	UINT16_4,

	// 32-bit
	INT32_1,
	INT32_2,
	INT32_3,
	INT32_4,

	UINT32_1,
	UINT32_2,
	UINT32_3,
	UINT32_4,

	// 64-bit
	INT64_1,
	INT64_2,
	INT64_3,
	INT64_4,

	UINT64_1,
	UINT64_2,
	UINT64_3,
	UINT64_4,
	
	// Normalized Data Types (DX UNORM or NORM)
	// Fixed Point
	// Definition of UNORM/NORM can be found here
	// http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx

	// 8-bit
	NORM8_1,
	NORM8_2,
	NORM8_3,
	NORM8_4,

	UNORM8_1,
	UNORM8_2,
	UNORM8_3,
	UNORM8_4,

	// 16-bit
	NORM16_1,
	NORM16_2,
	NORM16_3,
	NORM16_4,

	UNORM16_1,
	UNORM16_2,
	UNORM16_3,
	UNORM16_4,

	// 32-bit
	NORM32_1,
	NORM32_2,
	NORM32_3,
	NORM32_4,

	UNORM32_1,
	UNORM32_2,
	UNORM32_3,
	UNORM32_4,

	// Packed Data Types
	INT_2_10_10_10,		// Packed Data, LSB to MSB is 10 to 2, unpacked format is 4 normalized integers
	UINT_2_10_10_10,	// Packed Data, LSB to MSB is 10 to 2, unpacked format is 4 unsigned normalized integers

	// https://www.opengl.org/registry/specs/EXT/packed_float.txt
	UINT_10F_11F_11F,	// Packed Data, LSB to MSB is 11F to 10F, unpacked format is 3 floating point numbers

	//------------------------------------//
	//------------------------------------//
	// Custom Data Types
	// Generic normalized data packed
	CUSTOM_1_15N_16N,	// Packed Data, LSB to MSB is 16F to 15F, msb bit shows sign of the z component of the vector
						// 15F is Y component, 16F is the X component
						// X and Y components are fixed point [-1, 1]
						// Z = sqrt(X^2 + Y^2) * ((bit) ? -1 : 1)
						//
						// OGL Feed : use GL_UNSIGNED_INT with one component
						// Unpack it on shader

	// Tangent Binormal Packed Data
	CUSTOM_TANG_H_2N,	// Packed Data, LSB to MSB
						// 2N:  2 x norm32 types (for normalized tangent)
						// which shows x, y
						// H: 4 byte data in which
						// two byte shows binorm direction (least 
						// other two byte show tang direction
						// binorm headedness shows weather "N cross T"
						// is on wrong direction
						// positive dir (0x00), negative dir (0x01)
						// OGL Feed : use GL_FLOAT with three component
						// Unpack H portion on shader

	// High Amount Weight data
	UNORM8_4_4,			// 8 bit unorm weight in a 4 component 32 bit type
						// holds 16 8 bit unorm values
	UNORM16_2_4,		// 16 bit unorm weight in a 4 component 32 bit type
						// holds 8 16 bit unorm value
						//
						// OGL Feed : use GL_INT with four component
						// Unpack H portion on shader

	UINT8_4_4,			// 8 bit integer weight in a 4 component 32 bit type
						// holds 16 8 bit integer values
	UINT16_2_4,			// 16 bit integer index in a 4 component 32 bit type
						// holds 8 16 bit integer values
						// OGL Feed : use GL_INT with four component

	END					// For Static Asserting the size array
};

static std::ostream& operator<<(std::ostream& os, const GFGDataType& dt)
{
	static const char* values[] =
	{
		"HALF_1",
		"HALF_2",
		"HALF_3",
		"HALF_4",
		"FLOAT_1",
		"FLOAT_2",
		"FLOAT_3",
		"FLOAT_4",
		"DOUBLE_1",
		"DOUBLE_2",
		"DOUBLE_3",
		"DOUBLE_4",
		"QUADRUPLE_1",
		"QUADRUPLE_2",
		"QUADRUPLE_3",
		"QUADRUPLE_4",
		"INT8_1",
		"INT8_2",
		"INT8_3",
		"INT8_4",
		"UINT8_1",
		"UINT8_2",
		"UINT8_3",
		"UINT8_4",
		"INT16_1",
		"INT16_2",
		"INT16_3",
		"INT16_4",
		"UINT16_1",
		"UINT16_2",
		"UINT16_3",
		"UINT16_4",
		"INT32_1",
		"INT32_2",
		"INT32_3",
		"INT32_4",
		"UINT32_1",
		"UINT32_2",
		"UINT32_3",
		"UINT32_4",
		"INT64_1",
		"INT64_2",
		"INT64_3",
		"INT64_4",
		"UINT64_1",
		"UINT64_2",
		"UINT64_3",
		"UINT64_4",
		"NORM8_1",
		"NORM8_2",
		"NORM8_3",
		"NORM8_4",
		"UNORM8_1",
		"UNORM8_2",
		"UNORM8_3",
		"UNORM8_4",
		"NORM16_1",
		"NORM16_2",
		"NORM16_3",
		"NORM16_4",
		"UNORM16_1",
		"UNORM16_2",
		"UNORM16_3",
		"UNORM16_4",
		"NORM32_1",
		"NORM32_2",
		"NORM32_3",
		"NORM32_4",
		"UNORM32_1",
		"UNORM32_2",
		"UNORM32_3",
		"UNORM32_4",
		"INT_2_10_10_10",
		"UINT_2_10_10_10",
		"UINT_10F_11F_11F",
		"CUSTOM_1_15F_16F",
		"CUSTOM_TANG_2F_H",
		"UNORM8_4_4",
		"UNORM16_2_4",
		"UINT8_4_4",
		"UINT16_2_4",
		"END"
	};
	return os << values[static_cast<int>(dt)];
}

enum class GFGIndexDataType : uint32_t
{
	UINT8,
	UINT16,
	UINT32
};

enum class GFGVertexComponentLogic : uint32_t
{
	// Generic Data Types
	POSITION,
	UV,
	NORMAL,
	TANGENT,
	BINORMAL,
	WEIGHT,
	WEIGHT_INDEX,
	COLOR,

	//------------------------------------//
	//------------------------------------//
	// Custom Data Types
};

static const size_t GFGDataTypeByteSize[]
{
	// Float Data Types
	2 * 1,	// HALF_1,			// one component IEEE 754 (16-bit) floating point number
	2 * 2,	// HALF_2,			// two component                   ""
	2 * 3,	// HALF_3,			// three component                 ""
	2 * 4,	// HALF_4,			// four component			       ""

	4 * 1,	// FLOAT_1,			// one component IEEE 754 (32-bit) floating point number
	4 * 2,	// FLOAT_2,			// two component                   ""
	4 * 3,	// FLOAT_3,			// three component                 ""
	4 * 4,	// FLOAT_4,			// four component			       ""

	8 * 1,	// DOUBLE_1,		// one component IEEE 754 (64-bit) floating point number
	8 * 2,	// DOUBLE_2,		// two component			       ""
	8 * 3,	// DOUBLE_3,		// three component			       ""
	8 * 4,	// DOUBLE_4,		// four component			       ""

	// Nobody will ever us it in this decade maybe but w/e
	16 * 1,	// QUADRUPLE_1,		// one component IEEE 754 (128-bit) floating point number
	16 * 2,	// QUADRUPLE_2,		// two component			       ""
	16 * 3,	// QUADRUPLE_3,		// three component			       ""
	16 * 4,	// QUADRUPLE_4,		// four component			       ""

	// Classic Integers
	// 8-bit
	1 * 1,	// INT8_1,
	1 * 2,	// INT8_2,
	1 * 3,	// INT8_3,
	1 * 4,	// INT8_4,

	1 * 1,	// UINT8_1,
	1 * 2,	// UINT8_2,
	1 * 3,	// UINT8_3,
	1 * 4,	// UINT8_4,

	// 16-bit
	2 * 1,	// INT16_1,
	2 * 2,	// INT16_2,
	2 * 3,	// INT16_3,
	2 * 4,	// INT16_4,
			//
	2 * 1,	// UINT16_1,
	2 * 2,	// UINT16_2,
	2 * 3,	// UINT16_3,
	2 * 4,	// UINT16_4,

	// 32-bit
	4 * 1,	// INT32_1,
	4 * 2,	// INT32_2,
	4 * 3,	// INT32_3,
	4 * 4,	// INT32_4,

	4 * 1,	// UINT32_1,
	4 * 2,	// UINT32_2,
	4 * 3,	// UINT32_3,
	4 * 4,	// UINT32_4,

	// 64-bit
	8 * 1,	// INT64_1,
	8 * 2,	// INT64_2,
	8 * 3,	// INT64_3,
	8 * 4,	// INT64_4,

	8 * 1,	// UINT64_1,
	8 * 2,	// UINT64_2,
	8 * 3,	// UINT64_3,
	8 * 4,	// UINT64_4,

	// Normalized Data Types (DX UNORM or NORM)
	// Fixed Point
	// Definition of UNORM/NORM can be found here
	// http://msdn.microsoft.com/en-us/library/windows/desktop/dd607323(v=vs.85).aspx

	// 8-bit
	1 * 1,	// NORM8_1,
	1 * 2,	// NORM8_2,
	1 * 3,	// NORM8_3,
	1 * 4,	// NORM8_4,

	1 * 1,	// UNORM8_1,
	1 * 2,	// UNORM8_2,
	1 * 3,	// UNORM8_3,
	1 * 4,	// UNORM8_4,

	// 16-bit
	2 * 1,	// NORM16_1,
	2 * 2,	// NORM16_2,
	2 * 3,	// NORM16_3,
	2 * 4,	// NORM16_4,

	2 * 1,	// UNORM16_1,
	2 * 2,	// UNORM16_2,
	2 * 3,	// UNORM16_3,
	2 * 4,	// UNORM16_4,

	// 32-bit
	4 * 1,	// NORM32_1,
	4 * 2,	// NORM32_2,
	4 * 3,	// NORM32_3,
	4 * 4,	// NORM32_4,

	4 * 1,	// UNORM32_1,
	4 * 2,	// UNORM32_2,
	4 * 3,	// UNORM32_3,
	4 * 4,	// UNORM32_4,

	// Packed Data Types
	4 * 1,	// INT_2_10_10_10,		// Packed Data, LSB to MSB is 10 to 2, unpacked format is 4 integers
	4 * 2,	// UINT_2_10_10_10,		// Packed Data, LSB to MSB is 10 to 2, unpacked format is 4 unsigned integers
	
	// https://www.opengl.org/registry/specs/EXT/packed_float.txt
	4 * 1,	// UINT_10F_11F_11F		// Packed Data, LSB to MSB is 11F to 10F, unpacked format is 3 floating point numbers
	
	//------------------------------------//
	//------------------------------------//
	// Custom Data Types
	4,		// CUSTOM_1_15N_16N,		// Packed Data, LSB to MSB is 16F to 15F, msb bit shows sign of the z component of the vector
										// 15F is Y component, 16F is the X component
										// X and Y components are fixed point [-1, 1] (for easier coversion to float)
										// Z = sqrt(X^2 + Y^2) * ((bit) ? -1 : 1)
										//
										// OGL Feed : use GL_UNSIGNED_INT with one component
										// Unpack it on shader
	4 * 3,	//CUSTOM_TANG_H_2N,			// Packed Data, LSB to MSB
										// 12 byte composition of 2 floats (normalized tangent)
										// which shows x
										// 1 4 byte data in which one byte shows binorm direction
										// other bytes show tang direction

	// High Amount Weight data
	16,		//UNORM8_4_4,			// 8 bit unorm weight in a 4 component 32 bit type
									// holds 16 8 bit unorm values
	16,		//UNORM16_2_4,			// 16 bit unorm weight in a 4 component 32 bit type
									// holds 8 16 bit unorm values

	16,		//UINT8_4_4,			// 8 bit integer weight in a 4 component 32 bit type
									// holds 16 8 bit integer values
	16,		//UINT16_2_4,			// 16 bit integer index in a 4 component 32 bit type
									// holds 8 16 bit integer values
};

static_assert((sizeof(GFGDataTypeByteSize) / sizeof(size_t)) == static_cast<size_t>(GFGDataType::END),
			  "\'GFGDataType\' enum and its size array does not have same amount of elements.");

enum class GFGTopology : uint32_t
{
	TRIANGLE,
	TRIANGLE_STRIP,
	LINE,
	POINT,
};

enum class GFGDirection
{
	FROM_START,
	FROM_CURRENT,
	FROM_END
};

enum class GFGStringType : uint32_t
{
	EMPTY,
	ASCII,
	UTF8
};

enum class GFGMaterialLogic : uint32_t
{
	EMPTY,			// Empty Logic This Mateiral Definition used to partition mesh

	// Maya Exports Direct
	MAYA_LAMBERT,	// Lambert Shader as in Maya
	MAYA_PHONG,		// Phong Shader as in Maya
	MAYA_DX11		// DX11 (with plugin) shader as in Maya

	//TODO: Add Major Programs shaders also (like blender, 3dmax)
};
#endif //__GFG_ENUMERATIONS_H__