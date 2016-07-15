/**


*/
#ifndef __MESHBATCHNYRA_H__
#define __MESHBATCHNYRA_H__

#include "MeshBatch.h"

class MeshBatchNyra : public MeshBatch
{
	private:

	protected:

	public:
		// Constructors & Destructor
								MeshBatchNyra(const char* sceneFileName,
											  float minVoxSpan,
											  const Array32<size_t> maxVoxelCounts);

		// Static Files
		static const char*		nyraFileName;
		static size_t			nyraVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};

#endif //__MESHBATCHNYRA_H__