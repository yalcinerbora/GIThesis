#include "VoxelDebugVAO.h"
#include "GFG/GFGFileLoader.h"
#include "GLHeader.h"
#include "ThesisSolution.h"

CubeData VoxelDebugVAO::voxelCubeData = {0, 0};
const char* VoxelDebugVAO::cubeGFGFileName = "cube.gfg";

void VoxelDebugVAO::InitVoxelCube()
{
	std::ifstream stream(cubeGFGFileName, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader loader(&stlFileReader);

	GFGFileError e;
	if((e = loader.ValidateAndOpen()) != GFGFileError::OK)
	{
		assert(false);
	}

	glGenBuffers(1, &voxelCubeData.vertexBuffer);
	glGenBuffers(1, &voxelCubeData.indexBuffer);

	std::vector<uint8_t> vertexData(loader.MeshVertexDataSize(0));
	std::vector<uint8_t> indexData(loader.MeshIndexDataSize(0));
	loader.MeshVertexData(vertexData.data(), 0);
	loader.MeshIndexData(indexData.data(), 0);

	glBindBuffer(GL_COPY_WRITE_BUFFER, voxelCubeData.vertexBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, vertexData.size(), vertexData.data(), GL_DYNAMIC_DRAW);

	glBindBuffer(GL_COPY_WRITE_BUFFER, voxelCubeData.indexBuffer);
	glBufferData(GL_COPY_WRITE_BUFFER, indexData.size(), indexData.data(), GL_DYNAMIC_DRAW);

	voxelCubeData.indexCount = static_cast<GLuint>(loader.Header().meshes[0].headerCore.indexCount);
}

VoxelDebugVAO::VoxelDebugVAO(StructuredBuffer<VoxelData>& voxDataBuffer,
							 StructuredBuffer<VoxelRenderData>& voxRenderDataBuffer,
							 uint32_t voxelCount)
	: vaoId(0)
	, voxelCount(voxelCount)
{
	if(voxelCubeData.indexBuffer == 0 &&
	   voxelCubeData.vertexBuffer == 0)
	{
		InitVoxelCube();
	}

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);

	GLuint buffers[] = {voxelCubeData.vertexBuffer, 
						voxDataBuffer.getGLBuffer(),
						voxRenderDataBuffer.getGLBuffer()};
	GLintptr offsets[] = { 0, 0, 0 };
	GLsizei strides[] = { sizeof(float) * 3, sizeof(VoxelData), sizeof(VoxelRenderData) }; 

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, voxelCubeData.indexBuffer);

	glBindVertexBuffers(0, 3, buffers, offsets, strides);
	//glBindVertexBuffer(0, voxelCubeData.vertexBuffer, 0, sizeof(float) * 3);
	// Cube Pos
	glEnableVertexAttribArray(0);
	glVertexAttribFormat(0,
						 3,
						 GL_FLOAT,
						 GL_FALSE,
						 0);
	glVertexAttribBinding(0, 0);

	// VoxData
	glEnableVertexAttribArray(2);
	glVertexAttribFormat(2,
						 2,
						 GL_UNSIGNED_INT,
						 GL_FALSE,
						 0);
	glVertexAttribDivisor(2, 1);
	glVertexAttribBinding(2, 1);

	//// Vox Color 
	//glEnableVertexAttribArray(1);
	//glVertexAttribFormat(1,
	//					 4,
	//					 GL_UNSIGNED_BYTE,
	//					 GL_TRUE,
	//					 sizeof(IEVector3));
	//glVertexAttribDivisor(1, 1);
	//glVertexAttribBinding(1, 2);

	//// Vox Normal
	//glEnableVertexAttribArray(3);
	//glVertexAttribFormat(3,
	//					 3,
	//					 GL_FLOAT,
	//					 GL_FALSE,
	//					 0);
	//glVertexAttribDivisor(3, 1);
	//glVertexAttribBinding(3, 2);
}

VoxelDebugVAO::~VoxelDebugVAO()
{
	glDeleteVertexArrays(1, &vaoId);
}

void VoxelDebugVAO::Draw()
{
	glBindVertexArray(vaoId);
	glDrawElementsInstanced(GL_TRIANGLES,
							voxelCubeData.indexCount,
							GL_UNSIGNED_INT,
							nullptr,
							1);
	//glDrawElements(GL_TRIANGLES, voxelCubeData.indexCount,
	//			   GL_UNSIGNED_INT, nullptr);
}
