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

VoxelDebugVAO::VoxelDebugVAO(StructuredBuffer<VoxelNormPos>& voxNormPosBuffer,
							 StructuredBuffer<VoxelIds>& voxIdBuffer,
							 StructuredBuffer<VoxelColorData>& voxRenderDataBuffer)
	: vaoId(0)
{
	if(voxelCubeData.indexBuffer == 0 &&
	   voxelCubeData.vertexBuffer == 0)
	{
		InitVoxelCube();
	}

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);

	GLuint buffers[] = {voxelCubeData.vertexBuffer, 
						voxNormPosBuffer.getGLBuffer(),
						voxIdBuffer.getGLBuffer(),
						voxRenderDataBuffer.getGLBuffer()};
	GLintptr offsets[] = { 0, 0, 0, 0 };
	GLsizei strides[] = { sizeof(float) * 3,
						  sizeof(VoxelNormPos),
						  sizeof(VoxelIds), 
						  sizeof(VoxelColorData) };

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, voxelCubeData.indexBuffer);

	glBindVertexBuffers(0, 4, buffers, offsets, strides);
	// Cube Pos
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS,
						 3,
						 GL_FLOAT,
						 GL_FALSE,
						 0);
	glVertexAttribBinding(IN_POS, 0);

	// VoxNormPos
	glEnableVertexAttribArray(IN_VOX_NORM_POS);
	glVertexAttribIFormat(IN_VOX_NORM_POS,
						  2,
						  GL_UNSIGNED_INT,
						  0);
	glVertexAttribDivisor(IN_VOX_NORM_POS, 1);
	glVertexAttribBinding(IN_VOX_NORM_POS, 1);

	// VoxIds
	glEnableVertexAttribArray(IN_VOX_IDS);
	glVertexAttribIFormat(IN_VOX_IDS,
						  2,
						  GL_UNSIGNED_INT,
						  0);
	glVertexAttribDivisor(IN_VOX_IDS, 1);
	glVertexAttribBinding(IN_VOX_IDS, 2);

	// Vox Color 
	glEnableVertexAttribArray(IN_VOX_COLOR);
	glVertexAttribFormat(IN_VOX_COLOR,
						 4,
						 GL_UNSIGNED_BYTE,
						 GL_TRUE,
						 0);
	glVertexAttribDivisor(IN_VOX_COLOR, 1);
	glVertexAttribBinding(IN_VOX_COLOR, 3);
}

VoxelDebugVAO::VoxelDebugVAO(StructuredBuffer<VoxelNormPos>& voxNormPosBuffer,
							 StructuredBuffer<uchar4>& voxRenderDataBuffer)
	: vaoId(0)
{
	if(voxelCubeData.indexBuffer == 0 &&
	   voxelCubeData.vertexBuffer == 0)
	{
		InitVoxelCube();
	}

	glGenVertexArrays(1, &vaoId);
	glBindVertexArray(vaoId);

	GLuint buffers[] = {voxelCubeData.vertexBuffer,
						voxNormPosBuffer.getGLBuffer(),
						voxRenderDataBuffer.getGLBuffer()};
	GLintptr offsets[] = { 0, 0, 0 };
	GLsizei strides[] = { sizeof(float) * 3, 
						  sizeof(VoxelNormPos),
						  sizeof(uchar4) };

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, voxelCubeData.indexBuffer);

	glBindVertexBuffers(0, 3, buffers, offsets, strides);
	// Cube Pos
	glEnableVertexAttribArray(IN_POS);
	glVertexAttribFormat(IN_POS,
						 3,
						 GL_FLOAT,
						 GL_FALSE,
						 0);
	glVertexAttribBinding(IN_POS, 0);

	// VoxNormPos
	glEnableVertexAttribArray(IN_VOX_NORM_POS);
	glVertexAttribIFormat(IN_VOX_NORM_POS,
						  2,
						  GL_UNSIGNED_INT,
						  0);
	glVertexAttribDivisor(IN_VOX_NORM_POS, 1);
	glVertexAttribBinding(IN_VOX_NORM_POS, 1);

	// Vox Color 
	glEnableVertexAttribArray(IN_VOX_COLOR);
	glVertexAttribFormat(IN_VOX_COLOR,
						 4,
						 GL_UNSIGNED_BYTE,
						 GL_TRUE,
						 0);
	glVertexAttribDivisor(IN_VOX_COLOR, 1);
	glVertexAttribBinding(IN_VOX_COLOR, 2);
}

VoxelDebugVAO::VoxelDebugVAO(VoxelDebugVAO&& mv)
	: vaoId(mv.vaoId)
{
	mv.vaoId = 0;
}

VoxelDebugVAO::~VoxelDebugVAO()
{
	glDeleteVertexArrays(1, &vaoId);
}

void VoxelDebugVAO::Bind()
{
	glBindVertexArray(vaoId);
}

void VoxelDebugVAO::Draw(uint32_t voxelCount, uint32_t offset)
{
	glDrawElementsInstancedBaseInstance(GL_TRIANGLES,
										voxelCubeData.indexCount,
										GL_UNSIGNED_INT,
										nullptr,
										voxelCount,
										offset);
}
