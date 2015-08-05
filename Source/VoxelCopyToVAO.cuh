/**

Copying voxels to VAO

*/

#ifndef __VOXELCOPYTOVAO_H__
#define __VOXELCOPYTOVAO_H__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

struct CVoxelPage;
struct CVoxelGrid;
struct CObjectVoxelInfo;
struct CVoxelRender;

extern __device__ void DetermineTotalVoxCount(int& totalVox,
											  
											  // Per Obj Segment
											  const ushort2* gObjectAllocLocations,

											  // Per obj
											  const unsigned int* gObjectAllocIndexLookup,
											  const CObjectVoxelInfo* gObjInfo,
											  uint32_t objectCount);

extern __device__ void VoxelCopyToVAO(// Two ogl Buffers for rendering used voxels
									  uint4* voxelData,
									  uchar4* voxelColorData,
									  unsigned int& atomicIndex,

									  // Per Obj Segment
									  const ushort2** gObjectAllocLocations,

									  // Per obj
									  const unsigned int** gObjectAllocIndexLookup,

									  // Per vox
									  const CVoxelRender** gVoxelRenderData,

									  // Data
									  const CVoxelPage* gVoxPages,
									  uint32_t pageCount,
									  const CVoxelGrid& gGridInfo);

#endif //__THESISSOLUTION_H__