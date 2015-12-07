/**


*/
#ifndef __MESHBATCHCUBE_H__
#define __MESHBATCHCUBE_H__

#include "MeshBatchStatic.h"

class MeshBatchCube : public MeshBatchStatic
{
	private:

	protected:

	public:
		// Constructors & Destructor
								MeshBatchCube(const char* sceneFileName,
											  float minVoxSpan,
											  const Array32<size_t> maxVoxelCounts);

		// Static Files
		static const char*		rotatingCubeFileName;
		static size_t			rotatingCubeVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};

#endif //__MESHBATCHCUBE_H__