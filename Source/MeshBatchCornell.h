/**


*/
#ifndef __MESHBATCHCORNELL_H__
#define __MESHBATCHCORNELL_H__

#include "MeshBatch.h"

class MeshBatchCornell : public MeshBatch
{
	private:

	protected:

	public:
		// Constructors & Destructor
								MeshBatchCornell(const char* sceneFileName,
												 float minVoxSpan);

		// Static Files
		static const char*		cornellDynamicFileName;
		static size_t			cornellDynamicVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};

#endif //__MESHBATCHCORNELL_H__