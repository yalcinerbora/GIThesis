/**

Vao for rendering voxel debug

*/

#ifndef __VOXELDEBUGVAO_H__
#define __VOXELDEBUGVAO_H__

#include "GLHeaderLite.h"
#include <cstdint>
#include <vector_types.h>

struct VoxelData;
struct VoxelRenderData;

template<class T>
class StructuredBuffer;

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
								VoxelDebugVAO(StructuredBuffer<VoxelData>&,
											  StructuredBuffer<VoxelRenderData>&);
								VoxelDebugVAO(StructuredBuffer<VoxelData>&,
											  StructuredBuffer<uchar4>&);
								VoxelDebugVAO(const VoxelDebugVAO&) = delete;
		const VoxelDebugVAO&	operator= (const VoxelDebugVAO&) = delete;
								~VoxelDebugVAO();

		void					Draw(uint32_t voxelCount, uint32_t offset);
		void					Bind();


};
#endif //__VOXELDEBUGVAO_H__