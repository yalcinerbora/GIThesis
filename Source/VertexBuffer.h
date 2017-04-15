#pragma once
/**

GPU Buffer That Holds Mesh (and Index)

*/

#include <vector>
#include <map>
#include <cstdint>
#include "GLHeader.h"
#include "DrawPoint.h"
#include "StructuredBuffer.h"
#include "GFG/GFGEnumerations.h"

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

enum class VertexLogic
{
	POSITION,
	NORMAL,
	UV,
	WEIGHT,
	WEIGHT_INDEX
};

struct VertexElement
{	
	VertexLogic			logic;
	GPUDataType			type;
	uint32_t			typeCount;
	uint32_t			inputPosition;				
	uint32_t			offset;			
	bool				isNormInt;
};

class VertexBuffer
{
	private:		
		StructuredBuffer<uint8_t>				vertexBuffer;
		StructuredBuffer<uint8_t>				indexBuffer;
		GLuint									vao;

		uint32_t								byteStride;
		std::vector<size_t>						meshOffsets;
		const std::vector<VertexElement>		vElements;
		bool									addLocked;

		static bool								GFGLogicHit(GFGVertexComponentLogic, VertexLogic);
		static bool								GFGSupportedLogic(GFGVertexComponentLogic gfg);
		static bool								GFGSameDataType(GFGDataType, GPUDataType, uint32_t typeCount);
		void									GenerateVertexBuffer();
		
	protected:

	public:
		// Constructors & Destructor
									VertexBuffer();
									VertexBuffer(const std::vector<VertexElement>&,
												 uint32_t byteStride);
									VertexBuffer(const VertexBuffer&) = delete;
									VertexBuffer(VertexBuffer&&);
		VertexBuffer&				operator=(VertexBuffer&&);
		VertexBuffer&				operator=(const VertexBuffer&) = delete;
									~VertexBuffer();

		//
		void						AddMesh(DrawPointIndexed& result,
											const uint8_t data[],
											const uint8_t indexData[],
											size_t vertexCount,
											size_t indexCount);
		void						EditMesh(const uint8_t data[],
											 uint32_t meshId,
											 size_t vertexCount);
		
		bool						IsSuitedGFGMesh(const GFGMeshHeader&);

		void						Bind();
		void						LockAndLoad();
		void						AttachMTransformIndexBuffer(GLuint drawBuffer, size_t transformIndexOffset);

};