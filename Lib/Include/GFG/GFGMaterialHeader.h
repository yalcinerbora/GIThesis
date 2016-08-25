/**

GFGTexturePath Strcuture
GFGUniformData Structure
GFGMaterialHeaderCore Strcuture
GFGMaterialHeader Structure

Material Realted Structures used by GFGHeader

Various enumerations used by the GFGHeader.

For License refer to:
https://github.com/yalcinerbora/GFGFileFormat/blob/master/LICENSE
*/

#ifndef __GFG_MATERIALHEADER_H__
#define __GFG_MATERIALHEADER_H__

#include <vector>
#include "GFGEnumerations.h"

// Pack Those Structs 
#pragma pack(push, 1)


// Components of this material
struct GFGTexturePath
{
	uint64_t		stringLocation;		// String Location Relative to the texture Start (in bytes)
	GFGStringType	stringType;			// String Type (Only ASCII atm)
	uint32_t		stringSize;			// String Length (in bytes)
};

struct GFGUniformData
{
	uint64_t		dataLocation;		// String Location Relative to the uniform Start (in bytes)
	GFGDataType		dataType;			// Data Type
};

// Actual Core Header
struct GFGMaterialHeaderCore
{
	// Size Values
	uint32_t			textureCount;		// # of textures
	uint32_t			unifromCount;		// # of uniforms
	uint64_t			textureStart;		// Starting Location of Texture Data (byte offset relative to Data Start)
	uint64_t			uniformStart;		// Starting Location of Uniform Data (byte offset relative to Data Start)

	// Material Logic
	GFGMaterialLogic	logic;				// Enumeration that how this data should be interpreted
};

struct GFGMaterialHeader
{
	GFGMaterialHeaderCore		headerCore;
	std::vector<GFGTexturePath> textureList;
	std::vector<GFGUniformData> uniformList;
};

static_assert (sizeof(GFGMaterialHeaderCore) ==
			   sizeof(uint64_t) * 2
			   + sizeof(uint32_t) * 3,
			   "Material Header Core Size Mismatch from GFG Definition");

static_assert (sizeof(GFGTexturePath) ==
			   sizeof(uint64_t) * 1
			   + sizeof(uint32_t) * 2,
			   "Texture Path Size Mismatch from GFG Definition");

static_assert (sizeof(GFGUniformData) ==
			   sizeof(uint64_t) * 1
			   + sizeof(uint32_t) * 1,
			   "Uniform Data Size Mismatch from GFG Definition");

#pragma pack(pop)
#endif //__GFG_MATERIALHEADER_H__