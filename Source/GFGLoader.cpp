#include "GFGLoader.h"
#include "GFG/GFGFileLoader.h"
#include "GFG/GFGMaterialTypes.h"
#include "Macros.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"
#include "Material.h"

GFGLoadError GFGLoader::LoadGFG(GPUBuffer& buffer,
								DrawBuffer& glDrawBuffer,
								const char* gfgFilename)
{
	struct DrawCallMaterialPair
	{
		DrawPointIndexed drawCall;
		uint32_t materialIndex;
	};

	std::ifstream stream(gfgFilename, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);
	std::vector<DrawCallMaterialPair> materialDrawCall;
	std::vector<Material> materials;

	// Validate that the all vertex definitions are the same
	// and suits the VAO of the GPU Buffer
	uint64_t vertexCount = 0;
	for(const GFGMeshHeader mesh : gfgFile.Header().meshes)
	{
		if(!buffer.IsSuitedGFGMesh(mesh))
			return GFGLoadError::VAO_MISMATCH;
		vertexCount += mesh.headerCore.vertexCount;
	}
	if(!buffer.HasEnoughSpaceFor(vertexCount))
		return GFGLoadError::NOT_ENOUGH_SIZE;

	// Get All Mesh Vertex Data
	std::vector<uint8_t> vertexData;
	std::vector<uint8_t> indexData;
	vertexData.resize(gfgFile.AllMeshVertexDataSize());
	indexData.resize(gfgFile.AllMeshIndexDataSize());

	if(gfgFile.AllMeshVertexData(vertexData.data()) != GFGFileError::OK)
	{
		GI_ERROR_LOG("Failed to Load mesh vertex data on file %s", gfgFilename);
		return GFGLoadError::FATAL_ERROR;
	}
		
	if(gfgFile.AllMeshIndexData(indexData.data()) != GFGFileError::OK)
	{
		GI_ERROR_LOG("Failed to Load Mesh Index Data on file %s", gfgFilename);
		return GFGLoadError::FATAL_ERROR;
	}

	uint64_t vertexCount = 0;
	for(const GFGMeshHeader mesh : gfgFile.Header().meshes)
	{
		DrawPointIndexed drawCallParams;
		materialDrawCall.emplace_back();
		if(!buffer.AddMesh(materialDrawCall.back().drawCall, 
							vertexData.data() + (mesh.headerCore.vertexStart - gfgFile.Header().meshes[0].headerCore.vertexStart),
							indexData.data() + (mesh.headerCore.indexStart - gfgFile.Header().meshes[0].headerCore.indexStart),
							mesh.headerCore.vertexCount,
							mesh.headerCore.indexCount))
		{ 
			GI_ERROR_LOG("Failed to Load Mesh to GPU Buffer %s", gfgFilename);
			return GFGLoadError::FATAL_ERROR;
		}
	}


	for(const GFGMaterialHeader& mat : gfgFile.Header().materials)
	{
		// Get Color and Normal Material File Names

		ColorNormalMaterial material
		{
			new char[mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringSize],
			new char[mat.textureList[static_cast<int>(GFGMayaPhongLoc::)].stringSize]
		};
		mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringSize;



		Material m;


	}

	for(const GFGMeshMatPair& pair : gfgFile.Header().meshMaterialConnections.pairs)
	{
		pair.

	}



	return GFGLoadError::OK;
}