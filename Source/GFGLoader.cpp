#include "GFGLoader.h"
#include "GFG/GFGFileLoader.h"
#include "GFG/GFGMaterialTypes.h"
#include "Macros.h"
#include "GPUBuffer.h"
#include "DrawBuffer.h"
#include "Material.h"
#include "IEUtility/IEMatrix4x4.h"

GFGLoadError GFGLoader::LoadGFG(GPUBuffer& buffer,
								DrawBuffer& drawBuffer,
								const char* gfgFilename)
{
	std::ifstream stream(gfgFilename, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);
	std::vector<DrawPointIndexed> drawCalls;
	gfgFile.ValidateAndOpen();

	// Validate that the all vertex definitions are the same
	// and suits the VAO of the GPU Buffer
	uint64_t vertexCount = 0;
	uint64_t indexCount = 0;
	for(const GFGMeshHeader mesh : gfgFile.Header().meshes)
	{
		if(!buffer.IsSuitedGFGMesh(mesh))
			return GFGLoadError::VAO_MISMATCH;
		vertexCount += mesh.headerCore.vertexCount;
		indexCount += mesh.headerCore.indexCount;
	}
	if(!buffer.HasEnoughSpaceFor(vertexCount, indexCount))
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

	for(const GFGMeshHeader mesh : gfgFile.Header().meshes)
	{
		assert(mesh.headerCore.indexSize == sizeof(uint32_t));
		drawCalls.emplace_back();
		if(!buffer.AddMesh(drawCalls.back(),
							vertexData.data() + (mesh.headerCore.vertexStart - gfgFile.Header().meshes[0].headerCore.vertexStart),
							indexData.data() + (mesh.headerCore.indexStart - gfgFile.Header().meshes[0].headerCore.indexStart),
							mesh.headerCore.vertexCount,
							mesh.headerCore.indexCount))
		{ 
			GI_ERROR_LOG("Failed to Load Mesh to GPU Buffer %s", gfgFilename);
			return GFGLoadError::FATAL_ERROR;
		}
	}

	int matIndex = -1;
	for(const GFGMaterialHeader& mat : gfgFile.Header().materials)
	{
		matIndex++;
		// Get Color and Normal Material File Names
		std::vector<uint8_t> texNames;
		texNames.resize(gfgFile.MaterialTextureDataSize(matIndex));
		gfgFile.MaterialTextureData(texNames.data(), matIndex);

		ColorMaterial material;
		uint64_t start = mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringLocation;
		uint64_t end = start + mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringSize;
		assert(mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringType == GFGStringType::UTF8);
		material.colorFileName = reinterpret_cast<char*>(texNames.data() + start);
		drawBuffer.AddMaterial(material);
	}

	for(const GFGMeshMatPair& pair : gfgFile.Header().meshMaterialConnections.pairs)
	{
		DrawPointIndexed dpi = drawCalls[pair.meshIndex];
		dpi.firstIndex += static_cast<uint32_t>(pair.indexOffset);
		dpi.count = static_cast<uint32_t>(pair.indexCount);

		drawBuffer.AddDrawCall(dpi,
							   pair.materialIndex,
							   {IEMatrix4x4::IdentityMatrix,
								IEMatrix3x3::IdentityMatrix});
	}
	return GFGLoadError::OK;
}