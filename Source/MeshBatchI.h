/**


*/
#ifndef __MESHBATCHI_H__
#define __MESHBATCHI_H__

class DrawBuffer;
class GPUBuffer;
class SceneLights;

#include <cstdint>
#include <string>

enum class VoxelObjectType : uint32_t
{
	STATIC,			// Object does not move
	DYNAMIC,		// Object does move (with transform matrices)
	SKEL_DYNAMIC,	// Object moves with weighted transformation matrices
	MORPH_DYNAMIC,	// Object moves with morph targets (each voxel has their adjacent vertex morphs weighted)
};

class MeshBatchI
{
	public:
	// Interface
	virtual void				Update(double elapsedS) = 0;

	virtual DrawBuffer&			getDrawBuffer() = 0;
	virtual GPUBuffer&			getGPUBuffer() = 0;

	virtual const std::string&	BatchName() const = 0;

	virtual size_t				ObjectCount() const = 0;
	virtual size_t				DrawCount() const = 0;
	virtual size_t				MaterialCount() const = 0;
	virtual size_t				PolyCount() const = 0;

	virtual VoxelObjectType		MeshType() const = 0;
    virtual int                 RepeatCount() const = 0;

	virtual float				MinSpan() const = 0; // Minimum voxel span used in vox generation
	
};
#endif //__MESHBATCHI_H__