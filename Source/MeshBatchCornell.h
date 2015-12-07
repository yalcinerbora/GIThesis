/**


*/
#ifndef __MESHBATCHCORNELL_H__
#define __MESHBATCHCORNELL_H__

#include "MeshBatchStatic.h"

class MeshBatchCornell : public MeshBatchStatic
{
	private:

	protected:

	public:
		// Constructors & Destructor
								MeshBatchCornell(const char* sceneFileName,
												 float minVoxSpan,
												 const Array32<size_t> maxVoxelCounts);

		// Static Files
		static const char*		cornellDynamicFileName;
		static size_t			cornellDynamicVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};

#endif //__MESHBATCHCORNELL_H__