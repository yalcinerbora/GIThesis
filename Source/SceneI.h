/**

*/

#ifndef __SCENEI_H__
#define __SCENEI_H__

class DrawBuffer;
class GPUBuffer;

class SceneI 
{
	private:
	protected:
	public:
		virtual					~SceneI() = default;

		// Interface
		virtual DrawBuffer&		getDrawBuffer() = 0;
		virtual GPUBuffer&		getGPUBuffer() = 0;

		virtual size_t			ObjectCount() const = 0;
		virtual size_t			DrawCount() const = 0;
		virtual size_t			MaterialCount() const = 0;
		virtual size_t			PolyCount() const = 0;
};

#endif //__SCENEI_H__