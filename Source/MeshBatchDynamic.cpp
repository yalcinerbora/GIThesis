#include "MeshBatchDynamic.h"

// Constructors & Destructor
MeshBatchDynamic::MeshBatchDynamic(const char* sceneFileName,
								   float minVoxSpan,
								   const Array32<size_t> maxVoxelCounts,
								   BatchUpdateFunc func)
	: MeshBatchStatic(sceneFileName, minVoxSpan, maxVoxelCounts)
	, updateFunc(func)
{}

// Static Files
const char* MeshBatchDynamic::sponzaDynamicFileName = "sponzaDynamic.gfg";
const char* MeshBatchDynamic::cornellDynamicFileName = "cornellDynamic.gfg";
const char* MeshBatchDynamic::rotatingCubeFileName = "rainbowCube.gfg";

size_t MeshBatchDynamic::sponzaDynamicVoxelSizes[] =
{
	static_cast<size_t>(1024 * 30.0f),
	static_cast<size_t>(1024 * 10.0f),
	static_cast<size_t>(1024 * 5.0f)
};

size_t MeshBatchDynamic::cornellDynamicVoxelSizes[] =
{
	static_cast<size_t>(1024 * 1024 * 1.5f),
	static_cast<size_t>(1024 * 1024 * 2.0f),
	static_cast<size_t>(1024 * 1024 * 1.5f)
};

size_t MeshBatchDynamic::rotatingCubeVoxelSizes[] =
{
	static_cast<size_t>(1024 * 120.0f),
	static_cast<size_t>(1024 * 35.0f),
	static_cast<size_t>(1024 * 10.0f)
};

// Interface
void MeshBatchDynamic::Update(double elapsedS)
{
	updateFunc(batchVertex, batchDrawParams, elapsedS);
}