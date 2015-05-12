/**

Vao for rendering voxel debug

*/

#ifndef __VOXELDEBUGVAO_H__
#define __VOXELDEBUGVAO_H__

#include "GLHeaderLite.h"
#include <cstdint>

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
		uint32_t			voxelCount;

	protected:

	public:
		// Cosntructors & Destructor
								VoxelDebugVAO(StructuredBuffer<VoxelData>&,
											  StructuredBuffer<VoxelRenderData>&,
											  uint32_t voxelCount);
								VoxelDebugVAO(const VoxelDebugVAO&) = delete;
		const VoxelDebugVAO&	operator= (const VoxelDebugVAO&) = delete;
								~VoxelDebugVAO();

		void					Draw();


};
#endif //__VOXELDEBUGVAO_H__