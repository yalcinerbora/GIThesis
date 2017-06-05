#include "VoxelVAO.h"
#include "DrawPoint.h"
#include <GFG/GFGFileLoader.h>

VoxelVAO::VoxelVAO()
	: vao(0)
{}

VoxelVAO::VoxelVAO(StructuredBuffer<uint8_t>& buffer,
				   size_t cubePosOffset,
				   size_t voxPosOffset,
				   size_t voxNormOffset,
				   size_t voxAlbedoOffset,
				   size_t voxWeightOffset)
	: vao(0)
{
	GLuint bufferId = buffer.getGLBuffer();
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	GLuint buffers[] =
	{
		bufferId,
		bufferId,
		bufferId,
		bufferId,
		bufferId
	};
	GLintptr offsets[] = 
	{
		static_cast<GLintptr>(cubePosOffset),
		static_cast<GLintptr>(voxPosOffset),
		static_cast<GLintptr>(voxNormOffset),
		static_cast<GLintptr>(voxAlbedoOffset),
		static_cast<GLintptr>(voxWeightOffset)
	};
	GLsizei strides[] =
	{
		sizeof(float) * 3,
		sizeof(VoxelPosition),
		sizeof(VoxelNormal),
		sizeof(VoxelAlbedo),
		sizeof(VoxelWeights)
	};

	// Everything is on that single buffer
	GLsizei attributeCount = (voxWeightOffset == 0) ? 4 : 5;
	attributeCount = (voxAlbedoOffset == 0) ? 3 : attributeCount;
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferId);
	glBindVertexBuffers(0, attributeCount, buffers, offsets, strides);
	// Cube Pos
	glEnableVertexAttribArray(IN_CUBE_POS);
	glVertexAttribFormat(IN_CUBE_POS,
						 3,
						 GL_FLOAT,
						 GL_FALSE,
						 0);
	glVertexAttribBinding(IN_CUBE_POS, 0);
	
	// Voxel Pos
	glEnableVertexAttribArray(IN_VOXEL_POS);
	glVertexAttribIFormat(IN_VOXEL_POS,
						  1,
						  GL_UNSIGNED_INT,
						  0);
	glVertexAttribDivisor(IN_VOXEL_POS, 1);
	glVertexAttribBinding(IN_VOXEL_POS, 1);
	
	// Voxel Normal
	glEnableVertexAttribArray(IN_VOXEL_NORM);
	glVertexAttribIFormat(IN_VOXEL_NORM,
						  2,
						  GL_UNSIGNED_INT,
						  0);
	glVertexAttribDivisor(IN_VOXEL_NORM, 1);
	glVertexAttribBinding(IN_VOXEL_NORM, 2);
	
	// Vox Albedo
	if(voxAlbedoOffset != 0)
	{
		glEnableVertexAttribArray(IN_VOXEL_ALBEDO);
		glVertexAttribFormat(IN_VOXEL_ALBEDO,
							 4,
							 GL_UNSIGNED_BYTE,
							 GL_TRUE,
							 0);
		glVertexAttribDivisor(IN_VOXEL_ALBEDO, 1);
		glVertexAttribBinding(IN_VOXEL_ALBEDO, 3);
	}

	if(voxWeightOffset != 0)
	{
		glEnableVertexAttribArray(IN_VOXEL_WEIGHT);
		glVertexAttribIFormat(IN_VOXEL_WEIGHT,
							  2,
							  GL_UNSIGNED_INT,
							  0);
		glVertexAttribDivisor(IN_VOXEL_WEIGHT, 1);
		glVertexAttribBinding(IN_VOXEL_WEIGHT, 4);
	}
}

VoxelVAO::VoxelVAO(VoxelVAO&& other)
	: vao(other.vao)
{
	other.vao = 0;
}

VoxelVAO& VoxelVAO::operator=(VoxelVAO&& other)
{
	assert(&other != this);
	vao = other.vao;
	other.vao = 0;
	return *this;
}

VoxelVAO::~VoxelVAO()
{
	glDeleteVertexArrays(1, &vao);
}

void VoxelVAO::Bind()
{
	glBindVertexArray(vao);
}

void VoxelVAO::Draw(uint32_t cubeIndexSize,
					uint32_t voxelCount,
					uint32_t offset)
{
	glDrawElementsInstancedBaseInstance(GL_TRIANGLES,
										cubeIndexSize,
										GL_UNSIGNED_INT,
										nullptr,
										voxelCount,
										offset);
}

void VoxelVAO::Draw(uint32_t drawPointOffset)
{
	static_assert(sizeof(GLintptr) == sizeof(void*), "Unappropirate GL Offset Parameter");
	GLintptr offset = static_cast<GLintptr>(drawPointOffset);
	glDrawElementsIndirect(GL_TRIANGLES,
						   GL_UNSIGNED_INT,
						   (void *)(offset));
}

VoxelVAO::CubeOGL VoxelVAO::LoadCubeDataFromGFG()
{
	// Loading Cube (For rendering voxels)
	std::ifstream stream(CubeGFGFileName, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader loader(&stlFileReader);

	GFGFileError e = loader.ValidateAndOpen();
	assert(e == GFGFileError::OK);
	assert(loader.Header().meshes.size() == 1);

	CubeOGL cubeData;
	cubeData.data.resize(loader.MeshIndexDataSize(0) +
						 loader.MeshVertexDataSize(0));
	loader.MeshIndexData(cubeData.data.data(), 0);
	loader.MeshVertexData(cubeData.data.data() + loader.MeshIndexDataSize(0), 0);
	cubeData.drawCount = static_cast<GLuint>(loader.Header().meshes.front().headerCore.indexCount);
	return cubeData;
}



//#include "GFG/GFGFileLoader.h"
//#include "GLHeader.h"
//#include "ThesisSolution.h"
//
//CubeData VoxelDebugVAO::voxelCubeData = {0, 0};
//const char* VoxelDebugVAO::cubeGFGFileName = "cube.gfg";
//
//void VoxelDebugVAO::InitVoxelCube()
//{
//	std::ifstream stream(cubeGFGFileName, std::ios_base::in | std::ios_base::binary);
//	GFGFileReaderSTL stlFileReader(stream);
//	GFGFileLoader loader(&stlFileReader);
//
//	GFGFileError e;
//	if((e = loader.ValidateAndOpen()) != GFGFileError::OK)
//	{
//		assert(false);
//	}
//
//	glGenBuffers(1, &voxelCubeData.vertexBuffer);
//	glGenBuffers(1, &voxelCubeData.indexBuffer);
//
//	std::vector<uint8_t> vertexData(loader.MeshVertexDataSize(0));
//	std::vector<uint8_t> indexData(loader.MeshIndexDataSize(0));
//	loader.MeshVertexData(vertexData.data(), 0);
//	loader.MeshIndexData(indexData.data(), 0);
//
//	glBindBuffer(GL_COPY_WRITE_BUFFER, voxelCubeData.vertexBuffer);
//	glBufferData(GL_COPY_WRITE_BUFFER, vertexData.size(), vertexData.data(), GL_DYNAMIC_DRAW);
//
//	glBindBuffer(GL_COPY_WRITE_BUFFER, voxelCubeData.indexBuffer);
//	glBufferData(GL_COPY_WRITE_BUFFER, indexData.size(), indexData.data(), GL_DYNAMIC_DRAW);
//
//	voxelCubeData.indexCount = static_cast<GLuint>(loader.Header().meshes[0].headerCore.indexCount);
//}
//
//VoxelDebugVAO::VoxelDebugVAO(StructuredBuffer<VoxelNormPos>& voxNormPosBuffer,
//							 StructuredBuffer<VoxelIds>& voxIdBuffer,
//							 StructuredBuffer<VoxelColorData>& voxRenderDataBuffer,
//							 StructuredBuffer<VoxelWeightData>& voxWeightDataBuffer,
//							 bool isSkeletal)
//	: vaoId(0)
//{
//	if(voxelCubeData.indexBuffer == 0 &&
//	   voxelCubeData.vertexBuffer == 0)
//	{
//		InitVoxelCube();
//	}
//
//	glGenVertexArrays(1, &vaoId);
//	glBindVertexArray(vaoId);
//
//	GLuint buffers[] = {voxelCubeData.vertexBuffer, 
//						voxNormPosBuffer.getGLBuffer(),
//						voxIdBuffer.getGLBuffer(),
//						voxRenderDataBuffer.getGLBuffer(),
//						voxWeightDataBuffer.getGLBuffer()};
//	GLintptr offsets[] = {0, 0, 0, 0, 0};
//	GLsizei strides[] = { sizeof(float) * 3,
//						  sizeof(VoxelNormPos),
//						  sizeof(VoxelIds), 
//						  sizeof(VoxelColorData),
//						  sizeof(VoxelWeightData)};
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, voxelCubeData.indexBuffer);
//	glBindVertexBuffers(0, (isSkeletal) ? 5 : 4, buffers, offsets, strides);
//	// Cube Pos
//	glEnableVertexAttribArray(IN_POS);
//	glVertexAttribFormat(IN_POS,
//						 3,
//						 GL_FLOAT,
//						 GL_FALSE,
//						 0);
//	glVertexAttribBinding(IN_POS, 0);
//
//	// VoxNormPos
//	glEnableVertexAttribArray(IN_VOX_NORM_POS);
//	glVertexAttribIFormat(IN_VOX_NORM_POS,
//						  2,
//						  GL_UNSIGNED_INT,
//						  0);
//	glVertexAttribDivisor(IN_VOX_NORM_POS, 1);
//	glVertexAttribBinding(IN_VOX_NORM_POS, 1);
//
//	// VoxIds
//	glEnableVertexAttribArray(IN_VOX_IDS);
//	glVertexAttribIFormat(IN_VOX_IDS,
//						  2,
//						  GL_UNSIGNED_INT,
//						  0);
//	glVertexAttribDivisor(IN_VOX_IDS, 1);
//	glVertexAttribBinding(IN_VOX_IDS, 2);
//
//	// Vox Color 
//	glEnableVertexAttribArray(IN_VOX_COLOR);
//	glVertexAttribFormat(IN_VOX_COLOR,
//						 4,
//						 GL_UNSIGNED_BYTE,
//						 GL_TRUE,
//						 0);
//	glVertexAttribDivisor(IN_VOX_COLOR, 1);
//	glVertexAttribBinding(IN_VOX_COLOR, 3);
//
//	if(isSkeletal)
//	{
//		glEnableVertexAttribArray(IN_VOX_WEIGHT);
//		glVertexAttribIFormat(IN_VOX_WEIGHT,
//							  2,
//							  GL_UNSIGNED_INT,
//							  0);
//		glVertexAttribDivisor(IN_VOX_WEIGHT, 1);
//		glVertexAttribBinding(IN_VOX_WEIGHT, 4);
//	}
//}
//
//VoxelDebugVAO::VoxelDebugVAO(StructuredBuffer<VoxelNormPos>& voxNormPosBuffer,
//							 StructuredBuffer<uchar4>& voxRenderDataBuffer)
//	: vaoId(0)
//{
//	if(voxelCubeData.indexBuffer == 0 &&
//	   voxelCubeData.vertexBuffer == 0)
//	{
//		InitVoxelCube();
//	}
//
//	glGenVertexArrays(1, &vaoId);
//	glBindVertexArray(vaoId);
//
//	GLuint buffers[] = {voxelCubeData.vertexBuffer,
//						voxNormPosBuffer.getGLBuffer(),
//						voxRenderDataBuffer.getGLBuffer()};
//	GLintptr offsets[] = { 0, 0, 0 };
//	GLsizei strides[] = { sizeof(float) * 3, 
//						  sizeof(VoxelNormPos),
//						  sizeof(uchar4)};
//
//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, voxelCubeData.indexBuffer);
//
//	glBindVertexBuffers(0, 3, buffers, offsets, strides);
//	// Cube Pos
//	glEnableVertexAttribArray(IN_POS);
//	glVertexAttribFormat(IN_POS,
//						 3,
//						 GL_FLOAT,
//						 GL_FALSE,
//						 0);
//	glVertexAttribBinding(IN_POS, 0);
//
//	// VoxNormPos
//	glEnableVertexAttribArray(IN_VOX_NORM_POS);
//	glVertexAttribIFormat(IN_VOX_NORM_POS,
//						  2,
//						  GL_UNSIGNED_INT,
//						  0);
//	glVertexAttribDivisor(IN_VOX_NORM_POS, 1);
//	glVertexAttribBinding(IN_VOX_NORM_POS, 1);
//
//	// Vox Color 
//	glEnableVertexAttribArray(IN_VOX_COLOR);
//	glVertexAttribFormat(IN_VOX_COLOR,
//						 4,
//						 GL_UNSIGNED_BYTE,
//						 GL_TRUE,
//						 0);
//	glVertexAttribDivisor(IN_VOX_COLOR, 1);
//	glVertexAttribBinding(IN_VOX_COLOR, 2);
//}
//
//VoxelDebugVAO::VoxelDebugVAO(VoxelDebugVAO&& mv)
//	: vaoId(mv.vaoId)
//{
//	mv.vaoId = 0;
//}
//
//VoxelDebugVAO::~VoxelDebugVAO()
//{
//	glDeleteVertexArrays(1, &vaoId);
//}
//
//void VoxelDebugVAO::Bind()
//{
//	glBindVertexArray(vaoId);
//}
//
//void VoxelDebugVAO::Draw(uint32_t voxelCount, uint32_t offset)
//{
//	glDrawElementsInstancedBaseInstance(GL_TRIANGLES,
//										voxelCubeData.indexCount,
//										GL_UNSIGNED_INT,
//										nullptr,
//										voxelCount,
//										offset);
//}
