/**

*/

#ifndef __SCENEI_H__
#define __SCENEI_H__

class MeshBatchI;
class SceneLights;

#include <cstdint>
#include "ArrayStruct.h"

class SceneI 
{
	private:
	protected:
	public:
		virtual							~SceneI() = default;

		// Interface
		virtual Array32<MeshBatchI*>	getBatches() = 0;
		virtual SceneLights&			getSceneLights() = 0;
		
		virtual size_t					ObjectCount() const = 0;
		virtual size_t					DrawCount() const = 0;
		virtual size_t					MaterialCount() const = 0;
		virtual size_t					PolyCount() const = 0;

		virtual void					Update(double elapsedS) = 0;

		virtual const uint32_t*			SVOLevelSizes() const = 0;// Level Sizes
};

#endif //__SCENEI_H__