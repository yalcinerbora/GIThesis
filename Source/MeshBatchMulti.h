#pragma once
/**


*/

#include "MeshBatchSkeletal.h"

class MeshBatchMultiSkel : public MeshBatchSkeletal
{
    private:
        int                         repeatCount;

    protected:
    public:
                                    MeshBatchMultiSkel(const char* sceneFileName,
                                                       float minVoxSpan, 
                                                       float xStart, float zStart,
                                                       float distance,
                                                       float width,
                                                       int repeatCount);

        int                         RepeatCount() const override;

        static constexpr char*      nyraName = "nyraJump.gfg";
};

class MeshBatchMulti : public MeshBatch
{
    private:
        int                         repeatCount;

    protected:
    public:
                                    MeshBatchMulti(const char* sceneFileName,
                                                   float minVoxSpan,
                                                   float xStart, float zStart,
                                                   float distance,
                                                   float width,
                                                   int repeatCount);

        int                         RepeatCount() const override;

        static constexpr char*      testBallName = "testBall.gfg";
};