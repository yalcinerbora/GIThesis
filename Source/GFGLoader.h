/**

Helper Class that loads GFG directly to the buffer

*/

#ifndef __GFGLOADER_H__
#define __GFGLOADER_H__

#include "GFG/GFGFileLoader.h"
#include "Material.h"
#include "StructuredBuffer.h"

class GPUBuffer;
class DrawBuffer;
class Material;
struct BatchParams;

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
	GFGLoadError	LoadGFG(BatchParams& params,
							GPUBuffer& buffer,
							DrawBuffer& drawBuffer,
							const char* gfgFilename,
							bool isSkeletal);
	GFGLoadError	LoadAnim(StructuredBuffer<IEVector4>& animKeys,
							 StructuredBuffer<GFGTransform>& bindPose,
							 StructuredBuffer<uint32_t>& jointHier,
							 const char* gfgFileName);
};

#endif //__GFGLOADER_H__