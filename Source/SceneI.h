/**

*/

#ifndef __SCENEI_H__
#define __SCENEI_H__

class DrawBuffer;
class GPUBuffer;
class SceneLights;

#include <cstdint>

class SceneI 
{
	private:
	protected:
	public:
		virtual					~SceneI() = default;

		// Interface
		virtual DrawBuffer&		getDrawBuffer() = 0;
		virtual GPUBuffer&		getGPUBuffer() = 0;
		virtual SceneLights&	getSceneLights() = 0;

		virtual size_t			ObjectCount() const = 0;
		virtual size_t			DrawCount() const = 0;
		virtual size_t			MaterialCount() const = 0;
		virtual size_t			PolyCount() const = 0;

		virtual float			MinSpan() const = 0; // Minimum voxel span used in vox generation
		virtual uint32_t		SVOTotalSize() const = 0; // SVO Total Size Malloc
		virtual const uint32_t*	SVOLevelSizes() const = 0;// Level Sizes
};

#endif //__SCENEI_H__