/**

Helper Class that loads GFG directly to the buffer

*/

#ifndef __GFGLOADER_H__
#define __GFGLOADER_H__

#include "GFG/GFGFileLoader.h"
#include "GPUBuffer.h"
#include "Material.h"

enum class GFGLoadError
{
	OK,

	// Buffer Related
	VAO_MISMATCH,
	NOT_ENOUGH_SIZE,

	// Material Related
	TEXTURE_NOT_FOUND,	// Texture Path Does not relate to 



	FATAL_ERROR
};

namespace GFGLoader
{
	// Mesh Related
	GFGLoadError	LoadToBuffer(GPUBuffer& buffer,
								 const Array32<const char*> gfgFilenames);	// Load every mesh on the files specified
	GFGLoadError	LoadToBuffer(GPUBuffer& buffer,
								 const char* gfgFilename,
								 const Array32<uint32_t> meshIDs);			// Load Specific Meshes on a GFG File

	// Material Related
	GFGLoadError	LoadMaterial(Material& material,
								 const char* gfgFilename,
								 const Array32<uint32_t> materialIDs);

};

#endif //__GFGLOADER_H__