/**


*/
#ifndef __MESHBATCHI_H__
#define __MESHBATCHI_H__

class DrawBuffer;
class VertexBuffer;
class SceneLights;

#include <cstdint>
#include <string>

//enum class VoxelObjectType : uint32_t
//{
//	STATIC,			// Object does not move
//	DYNAMIC,		// Object does move (with transform matrices)
//	SKEL_DYNAMIC,	// Object moves with weighted transformation matrices
//	MORPH_DYNAMIC,	// Object moves with morph targets (each voxel has their adjacent vertex morphs weighted)
//};

enum class MeshBatchType
{
	RIGID,
	SKELETAL,
	END
};
static constexpr uint32_t MeshBatchTypeCount = static_cast<uint32_t>(MeshBatchType::END);

class MeshBatchI
{
	public:
	// Interface
	virtual void				Update(double elapsedS) = 0;

	virtual DrawBuffer&			getDrawBuffer() = 0;
	virtual VertexBuffer&		getVertexBuffer() = 0;

	virtual size_t				ObjectCount() const = 0;
	virtual size_t				DrawCount() const = 0;
	virtual size_t				MaterialCount() const = 0;
	virtual size_t				PolyCount() const = 0;

	virtual MeshBatchType		MeshType() const = 0;
    virtual int                 RepeatCount() const = 0;	
};
#endif //__MESHBATCHI_H__