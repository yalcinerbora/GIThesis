/**

Vao for rendering voxel debug

*/

#ifndef __VOXELDEBUGVAO_H__
#define __VOXELDEBUGVAO_H__

#include "GLHeaderLite.h"
#include <cstdint>
#include <vector_types.h>
#include <array>
#include "VoxelCacheData.h"
#include "StructuredBuffer.h"

#define IN_POS 0
#define IN_VOX_COLOR 1
#define IN_VOX_NORM_POS 2
#define IN_VOX_IDS 3
#define IN_VOX_WEIGHT 4

class VoxelCacheBatch {public: struct Offsets; };

class VoxelDebugVAO
{
	private:
		GLuint					vao;

	public:
		// Cosntructors & Destructor
								VoxelDebugVAO(StructuredBuffer<uint8_t>&,
											  const VoxelCacheBatch::Offsets& offsets,
											  bool isSkeletal);
								VoxelDebugVAO(StructuredBuffer<VoxelNormPos>&,
											  StructuredBuffer<uchar4>&);
								VoxelDebugVAO(const VoxelDebugVAO&) = delete;
								VoxelDebugVAO(VoxelDebugVAO&&);
		VoxelDebugVAO&			operator=(const VoxelDebugVAO&) = delete;
		VoxelDebugVAO&			operator=(VoxelDebugVAO&&);
								~VoxelDebugVAO();

		void					Draw(uint32_t voxelCount, uint32_t offset);
		void					Bind();
};
#endif //__VOXELDEBUGVAO_H__