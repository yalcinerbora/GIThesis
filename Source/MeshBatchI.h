/**


*/
#ifndef __MESHBATCHI_H__
#define __MESHBATCHI_H__

class DrawBuffer;
class GPUBuffer;
class SceneLights;

#include <cstdint>

class MeshBatchI
{
	public:
	// Interface
	virtual void			Update(double elapsedS) = 0;

	virtual DrawBuffer&		getDrawBuffer() = 0;
	virtual GPUBuffer&		getGPUBuffer() = 0;

	virtual size_t			ObjectCount() const = 0;
	virtual size_t			DrawCount() const = 0;
	virtual size_t			MaterialCount() const = 0;
	virtual size_t			PolyCount() const = 0;

	virtual size_t			VoxelCacheMax(uint32_t level) const = 0;

	virtual float			MinSpan() const = 0; // Minimum voxel span used in vox generation
	
};

#endif //__MESHBATCHI_H__