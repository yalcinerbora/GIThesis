#pragma once
/**


*/

#include "MeshBatch.h"

class MeshBatchDyno : public MeshBatch
{
    private:

    protected:

    public:
        // Constructors & Destructor
                                 MeshBatchDyno(const char* sceneFileName,
                                               float minVoxSpan);

        // Interface
        void					Update(double elapsedS) override;
        VoxelObjectType			MeshType() const override;
};