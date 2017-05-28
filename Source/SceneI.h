/**

*/

#ifndef __SCENEI_H__
#define __SCENEI_H__

class MeshBatchI;
class SceneLights;

#include <vector>
#include <cstdint>

class SceneI 
{
	private:
	protected:
	public:
		virtual									~SceneI() = default;

		// Interface
		virtual const std::vector<std::string>&	getBatchFileNames(uint32_t batchId) = 0;
		virtual const std::vector<MeshBatchI*>&	getBatches() = 0;
		virtual SceneLights&					getSceneLights() = 0;
		virtual const SceneLights&				getSceneLights() const = 0;
	
		virtual size_t							ObjectCount() const = 0;
		virtual size_t							DrawCount() const = 0;
		virtual size_t							MaterialCount() const = 0;
		virtual size_t							PolyCount() const = 0;

		virtual void							Initialize() = 0;
		virtual void							Update(double elapsedS) = 0;
		virtual void							Load() = 0;
		virtual void							Release() = 0;

		virtual const std::string&				Name() const = 0;
};
#endif //__SCENEI_H__