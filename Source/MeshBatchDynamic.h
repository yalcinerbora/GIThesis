/**


*/
#ifndef __MESHBATCHDYNAMIC_H__
#define __MESHBATCHDYNAMIC_H__

#include "MeshBatchStatic.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"

typedef void(*BatchUpdateFunc)(GPUBuffer&, DrawBuffer&, double);

class MeshBatchDynamic : public MeshBatchStatic
{
	private:
		BatchUpdateFunc			updateFunc;

	protected:

	public:
		// Constructors & Destructor
								MeshBatchDynamic(const char* sceneFileName,
												 float minVoxSpan,
												 const Array32<size_t> maxVoxelCounts,
												 BatchUpdateFunc func);

		// Static Files
		static const char*		sponzaDynamicFileName;
		static const char*		cornellDynamicFileName;
		static const char*		rotatingCubeFileName;

		static size_t			sponzaDynamicVoxelSizes[];
		static size_t			cornellDynamicVoxelSizes[];
		static size_t			rotatingCubeVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};

#endif //__MESHBATCHDYNAMIC_H__