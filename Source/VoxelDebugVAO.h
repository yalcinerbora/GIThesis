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

//template<class T>
//class StructuredBuffer;
//

struct CubeData
{
	GLuint vertexBuffer;
	GLuint indexBuffer;

	GLuint indexCount;
};

class VoxelDebugVAO
{
	private:
		static CubeData		voxelCubeData;
		static const char*	cubeGFGFileName;

		void				InitVoxelCube();

		GLuint				vaoId;

	protected:

	public:
		// Cosntructors & Destructor
								VoxelDebugVAO(StructuredBuffer<VoxelNormPos>&,
											  StructuredBuffer<VoxelIds>&,
											  StructuredBuffer<VoxelColorData>&);
								VoxelDebugVAO(StructuredBuffer<VoxelNormPos>&,
											  StructuredBuffer<uchar4>&);
								VoxelDebugVAO(const VoxelDebugVAO&) = delete;
								VoxelDebugVAO(VoxelDebugVAO&&);
		const VoxelDebugVAO&	operator= (const VoxelDebugVAO&) = delete;
								~VoxelDebugVAO();

		void					Draw(uint32_t voxelCount, uint32_t offset);
		void					Bind();


};
#endif //__VOXELDEBUGVAO_H__