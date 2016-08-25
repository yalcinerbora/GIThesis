/**


*/
#ifndef __MESHBATCHSPONZA_H__
#define __MESHBATCHSPONZA_H__

#include "MeshBatch.h"

class MeshBatchSponza : public MeshBatch
{
	private:
	
	protected:

	public:
		// Constructors & Destructor
								MeshBatchSponza(const char* sceneFileName,
												float minVoxSpan);

		// Static Files
		static const char*		sponzaDynamicFileName;
		static size_t			sponzaDynamicVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};
#endif //__MESHBATCHSPONZA_H__