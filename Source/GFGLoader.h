/**

Helper Class that loads GFG directly to the buffer

*/

#ifndef __GFGLOADER_H__
#define __GFGLOADER_H__

#include "GFG/GFGFileLoader.h"
#include "Material.h"

class GPUBuffer;
class DrawBuffer;
class Material;
struct SceneParams;

enum class GFGLoadError
{
	OK,

	// Buffer Related
	VAO_MISMATCH,
	NOT_ENOUGH_SIZE,

	// Material Related
	TEXTURE_NOT_FOUND,
	FATAL_ERROR
};

namespace GFGLoader
{
	// Mesh Related
	GFGLoadError	LoadGFG(SceneParams& params,
							GPUBuffer& buffer,
							DrawBuffer& drawBuffer,
							const char* gfgFilename);

	//Material		GFGToMaterial(const GFGMaterialHeader& mat, );
};

#endif //__GFGLOADER_H__