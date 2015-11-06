/**

Copying voxels to VAO

*/

#ifndef __VOXELCOPYTOVAO_H__
#define __VOXELCOPYTOVAO_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

struct CVoxelPage;
struct CVoxelRender;
typedef uint2 CVoxelNormPos;
struct CVoxelGrid;
struct CObjectVoxelInfo;
struct CVoxelGrid;
struct CObjectTransform;

// Determine Vox count in pages
// Call Logic per page segment
extern __global__ void DetermineTotalVoxCount(int& totalVox,

											  const CVoxelPage* gVoxPages,
											  const CVoxelGrid& gGridInfo,
											  const uint32_t pageCount);


extern __global__ void VoxelCopyToVAO(// Two ogl Buffers for rendering used voxels
									  CVoxelNormPos* voxelData,
									  uchar4* voxelColorData,
									  unsigned int& atomicIndex,
									  const unsigned int maxBufferSize,

									  // Per Obj Segment
									  ushort2** gObjectAllocLocations,

									  // Per obj
									  unsigned int** gObjectAllocIndexLookup,

									  // Per vox
									  CVoxelRender** gVoxelRenderData,

									  // Page
									  const CVoxelPage* gVoxPages,
									  uint32_t pageCount,
									  const CVoxelGrid& gGridInfo,
									  
									  // Misc
									  bool isOuterCascade);

#endif //__THESISSOLUTION_H__