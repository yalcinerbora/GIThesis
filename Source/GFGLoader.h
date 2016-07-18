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
	GFGLoadError	LoadAnim(GFGAnimationHeader& header,
							 std::vector<IEVector3>& hipTranslations,
							 std::vector<float>& keyTimes,
							 std::vector<std::vector<IEQuaternion>>& jointKeys,
							 std::vector<GFGTransform>& bindPose,
							 std::vector<uint32_t>& jointHierarchy,
							 const char* gfgFileName);
};

#endif //__GFGLOADER_H__