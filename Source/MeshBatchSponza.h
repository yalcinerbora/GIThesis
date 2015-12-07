/**


*/
#ifndef __MESHBATCHSPONZA_H__
#define __MESHBATCHSPONZA_H__

#include "MeshBatchStatic.h"

class MeshBatchSponza : public MeshBatchStatic
{
	private:
	
	protected:

	public:
		// Constructors & Destructor
								MeshBatchSponza(const char* sceneFileName,
												float minVoxSpan,
												const Array32<size_t> maxVoxelCounts);

		// Static Files
		static const char*		sponzaDynamicFileName;
		static size_t			sponzaDynamicVoxelSizes[];

		// Interface
		void					Update(double elapsedS) override;
		VoxelObjectType			MeshType() const override;
};
#endif //__MESHBATCHSPONZA_H__