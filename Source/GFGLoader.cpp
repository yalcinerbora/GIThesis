#include "GFGLoader.h"
#include "GFG/GFGFileLoader.h"
#include "GFG/GFGMaterialTypes.h"
#include "Macros.h"
#include "MeshBatch.h"
#include "Material.h"
#include "IEUtility/IEMatrix4x4.h"
#include "Scene.h"
#include "Globals.h"

GFGLoadError GFGLoader::LoadGFG(BatchParams& params,
								GPUBuffer& buffer,
								DrawBuffer& drawBuffer,
								const char* gfgFileName,
								bool isSkeletal)
{
	std::ifstream stream(gfgFileName, std::ios_base::in | std::ios_base::binary);
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
		if(isSkeletal && !buffer.IsSuitedGFGMeshSkeletal(mesh))
			return GFGLoadError::VAO_MISMATCH;
		else if(!isSkeletal && !buffer.IsSuitedGFGMesh(mesh))
			return GFGLoadError::VAO_MISMATCH;

		vertexCount += mesh.headerCore.vertexCount;
		indexCount += mesh.headerCore.indexCount;
	}
	if(!buffer.HasEnoughSpaceFor(vertexCount, indexCount))
		return GFGLoadError::NOT_ENOUGH_SIZE;
	params.totalPolygons = indexCount / 3;
	

	// Get All Mesh Vertex Data
	std::vector<uint8_t> vertexData;
	std::vector<uint8_t> indexData;
	vertexData.resize(gfgFile.AllMeshVertexDataSize());
	indexData.resize(gfgFile.AllMeshIndexDataSize());
	if(gfgFile.AllMeshVertexData(vertexData.data()) != GFGFileError::OK)
	{
		GI_ERROR_LOG("Failed to Load mesh vertex data on file %s", gfgFileName);
		return GFGLoadError::FATAL_ERROR;
	}
		
	if(gfgFile.AllMeshIndexData(indexData.data()) != GFGFileError::OK)
	{
		GI_ERROR_LOG("Failed to Load Mesh Index Data on file %s", gfgFileName);
		return GFGLoadError::FATAL_ERROR;
	}

	params.objectCount = 0;
	for(const GFGMeshHeader mesh : gfgFile.Header().meshes)
	{
		assert(mesh.headerCore.indexSize == sizeof(uint32_t));
		drawCalls.emplace_back();
		if(!buffer.AddMesh(drawCalls.back(),
							vertexData.data() + (mesh.headerCore.vertexStart - gfgFile.Header().meshes[0].headerCore.vertexStart),
							indexData.data() + (mesh.headerCore.indexStart - gfgFile.Header().meshes[0].headerCore.indexStart),
							isSkeletal ? sizeof(VAOSkel) : sizeof(VAO),
							mesh.headerCore.vertexCount,
							mesh.headerCore.indexCount))
		{ 
			GI_ERROR_LOG("Failed to Load Mesh to GPU Buffer %s", gfgFileName);
			return GFGLoadError::FATAL_ERROR;
		}
		params.objectCount++;
	}
	
	int matIndex = -1;
	for(const GFGMaterialHeader& mat : gfgFile.Header().materials)
	{
		matIndex++;

		assert(mat.headerCore.logic == GFGMaterialLogic::MAYA_PHONG);
		assert(gfgFile.MaterialTextureDataSize(matIndex) > 0);

		ColorMaterial material;	
		std::vector<uint8_t> texNames;
		texNames.resize(gfgFile.MaterialTextureDataSize(matIndex));
		gfgFile.MaterialTextureData(texNames.data(), matIndex);
		if(mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringSize > 0)
		{
			uint64_t start = mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringLocation;
			uint64_t end = start + mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringSize;
			assert(mat.textureList[static_cast<int>(GFGMayaPhongLoc::COLOR)].stringType == GFGStringType::UTF8);
			material.colorFileName = reinterpret_cast<char*>(texNames.data() + start);
		}
			
		//if(mat.textureList[static_cast<int>(GFGMayaPhongLoc::BUMP)].stringSize > 0)
		//{
		//	uint64_t start = mat.textureList[static_cast<int>(GFGMayaPhongLoc::BUMP)].stringLocation;
		//	uint64_t end = start + mat.textureList[static_cast<int>(GFGMayaPhongLoc::BUMP)].stringSize;
		//	assert(mat.textureList[static_cast<int>(GFGMayaPhongLoc::BUMP)].stringType == GFGStringType::UTF8);
		//	material.normalFileName = reinterpret_cast<char*>(texNames.data() + start);
		//}
		drawBuffer.AddMaterial(material);
		params.materialCount++;
	}

	// Write Total Transform
	uint32_t transformIndex = 0;
	for(const GFGTransform& transform : gfgFile.Header().transformData.transforms)
	{
		auto HasThisTransform = [&transformIndex](const GFGNode& node) { return node.transformIndex == transformIndex; };
		const GFGNode* parent = &(*std::find_if(gfgFile.Header().sceneHierarchy.nodes.begin(),
												gfgFile.Header().sceneHierarchy.nodes.end(),
												HasThisTransform));
		IEMatrix4x4 transform = IEMatrix4x4::IdentityMatrix;
		IEMatrix3x3 transformRotation = IEMatrix3x3::IdentityMatrix;
		while(parent->parentIndex != -1)
		{
			const GFGTransform& t = gfgFile.Header().transformData.transforms[parent->transformIndex];

			transform = IEMatrix4x4::Rotate(t.rotate[0], IEVector3::Xaxis) * transform;
			transform = IEMatrix4x4::Rotate(t.rotate[1], IEVector3::Yaxis) * transform;
			transform = IEMatrix4x4::Rotate(t.rotate[2], IEVector3::Zaxis) * transform;

			transformRotation = transform;

			transform = IEMatrix4x4::Scale(t.scale[0], t.scale[1], t.scale[2]) * transform;
			transform = IEMatrix4x4::Translate({t.translate[0], t.translate[1], t.translate[2]}) * transform;

			parent = &gfgFile.Header().sceneHierarchy.nodes[parent->parentIndex];
		}

		drawBuffer.AddTransform(ModelTransform{transform, transformRotation});
		transformIndex++;
	}

	params.drawCallCount = 0;
	for(const GFGMeshMatPair& pair : gfgFile.Header().meshMaterialConnections.pairs)
	{
		DrawPointIndexed dpi = drawCalls[pair.meshIndex];
		dpi.firstIndex += static_cast<uint32_t>(pair.indexOffset);
		dpi.count = static_cast<uint32_t>(pair.indexCount);
		dpi.baseInstance = static_cast<uint32_t>(params.drawCallCount);

		uint32_t meshIndex = pair.meshIndex;
		auto FindMeshTransform = [&meshIndex](const GFGNode& node) { return node.meshReference == meshIndex; };
		uint32_t transformIndex = std::find_if(gfgFile.Header().sceneHierarchy.nodes.begin(),
											   gfgFile.Header().sceneHierarchy.nodes.end(),
											   FindMeshTransform)->transformIndex;
		drawBuffer.AddDrawCall
		(
			dpi,
			pair.materialIndex,
			transformIndex,
			{ 
				IEVector4(IEVector3(gfgFile.Header().meshes[pair.meshIndex].headerCore.aabb.min)),
				IEVector4(IEVector3(gfgFile.Header().meshes[pair.meshIndex].headerCore.aabb.max)),
			}
		);
		params.drawCallCount++;
	}
	buffer.AttachMTransformIndexBuffer(drawBuffer.getModelTransformIndexBuffer().getGLBuffer());
	return GFGLoadError::OK;
}

GFGLoadError GFGLoader::LoadAnim(StructuredBuffer<IEVector4>& animKeys,
								 StructuredBuffer<GFGTransform>& bindPose,
								 StructuredBuffer<uint32_t>& jointHier,
								 const char* gfgFileName)
{
	std::ifstream stream(gfgFileName, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);
	gfgFile.ValidateAndOpen();

	return GFGLoadError::OK;
}