/**

GPU Buffer That Holds Mesh (and Index)

*/

#ifndef __GPUBUFFER_H__
#define __GPUBUFFER_H__

#include <vector>
#include <map>
#include <cstdint>
#include "GLHeader.h"
#include "ArrayStruct.h"
#include "DrawPoint.h"

struct GFGMeshHeader;

enum class GPUDataType : GLenum
{
	INT8 = GL_BYTE,
	INT16 = GL_SHORT,
	INT32 = GL_INT,

	UINT8 = GL_UNSIGNED_BYTE,
	UINT16 = GL_UNSIGNED_SHORT,
	UINT32 = GL_UNSIGNED_INT,

	FLOAT = GL_FLOAT,
	DOUBLE = GL_DOUBLE
};

struct VertexElement
{
	uint32_t			inputPosition;		
	GPUDataType			type;
	bool				isNormInt;
	uint32_t			typeCount;			
	uint32_t			offset;				
	uint32_t			stride;				
};

class GPUBuffer
{
	private:
		static uint32_t				totalVertexCount;
		static uint32_t				totalIndexCount;
		
		GLuint						vertexBuffer;
		GLuint						indexBuffer;

		GLuint						vao;
		uint32_t					usedVertexAmount;
		uint32_t					usedIndexAmount;
		uint32_t					meshCount;

		std::vector<VertexElement>	vElements;

	protected:

	public:
		// Constructors & Destructor
									GPUBuffer(const Array32<const VertexElement>);
									GPUBuffer(const GPUBuffer&) = delete;
		const GPUBuffer&			operator=(const GPUBuffer&) = delete;
									~GPUBuffer();

		//
		bool						AddMesh(DrawPointIndexed& result,
											const uint8_t data[],
											const uint8_t indexData[],
											size_t vertexStride,
											size_t vertexCount,
											size_t indexCount);
		
		bool						IsSuitedGFGMesh(const GFGMeshHeader&);
		bool						IsSuitedGFGMeshSkeletal(const GFGMeshHeader&);
		bool						HasEnoughSpaceFor(uint64_t vertexCount,
													  uint64_t indexCount);

		void						Bind();
		void						AttachMTransformIndexBuffer(GLuint transformIndexBuffer);

};
#endif //__GPUBUFFER_H__