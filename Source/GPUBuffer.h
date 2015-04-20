/**

GPU Buffer That Holds Mesh (and Index)

Draw Points (Aka Draw Call Parameters) in that buffer
Draw Points Struct Applicable to the Indirect Command Buffer in OGL

*/

#ifndef __GPUBUFFER_H__
#define __GPUBUFFER_H__

#include <map>
#include <cstdint>
#include "GLHeader.h"
#include "ArrayStruct.h"
#include "GFG/GFGMeshHeader.h"

//struct DrawPoint
//{
//	uint32_t	count;
//	uint32_t	instanceCount;
//	uint32_t	baseVertex;
//	uint32_t	baseInstance;
//};

struct DrawPointIndexed
{
	uint32_t	count;
	uint32_t	instanceCount;
	uint32_t	firstIndex;
	uint32_t	baseVertex;
	uint32_t	baseInstance;
};

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

		std::vector<VertexElement>	vElements;

	protected:

	public:
		// Constructors & Destructor
									GPUBuffer(const Array32<VertexElement>);
									GPUBuffer(const GPUBuffer&) = delete;
		const GPUBuffer&			operator= (const GPUBuffer&) = delete;
									~GPUBuffer();

		//
		bool						AddMesh(DrawPointIndexed& result,
											const uint8_t data[],
											const uint8_t indexData[],
											size_t vertexCount,
											size_t indexCount);
		
		bool						IsSuitedGFGMesh(const GFGMeshHeader &);
		bool						HasEnoughSpaceFor(uint64_t vertexCount,
													  uint64_t indexCount);

		void						Bind();

};
#endif //__GPUBUFFER_H__