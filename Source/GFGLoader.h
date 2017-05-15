/**

Helper Class that loads GFG directly to the buffer

*/

#ifndef __GFGLOADER_H__
#define __GFGLOADER_H__

#include "GFG/GFGFileLoader.h"
#include "Material.h"
#include "StructuredBuffer.h"
#include "IEUtility/IEQuaternion.h"
#include "IEUtility/IEMatrix4x4.h"

class VertexBuffer;
class DrawBuffer;
class Material;
struct BatchParams;
class AnimationBatch;

enum class GFGLoadError
{
	OK,

	// Buffer Related
	VAO_MISMATCH,

	// Material Related
	TEXTURE_NOT_FOUND,
	FATAL_ERROR
};

namespace GFGLoader
{
	// Mesh Related
	GFGLoadError	LoadGFG(BatchParams& params,
							VertexBuffer& buffer,
							DrawBuffer& drawBuffer,
							const std::string& gfgFilename);
	GFGLoadError	LoadAnim(AnimationBatch& animationBatch,
							 const char* gfgFileName);
};

#endif //__GFGLOADER_H__