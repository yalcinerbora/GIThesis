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

			IEMatrix4x4 trans, rot;
			MeshBatch::GenTransformMatrix(trans, rot, t);

			transform = transform * trans;
			transformRotation = transformRotation * rot;

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

GFGLoadError GFGLoader::LoadAnim(GFGAnimationHeader& header,
								 std::vector<IEVector3>& hipTranslations,
								 std::vector<float>& keyTimes,
								 std::vector<std::vector<IEQuaternion>>& jointKeys,
								 std::vector<GFGTransform>& bindPose,
								 std::vector<uint32_t>& jointHierarchy,
								 const char* gfgFileName)
{
	std::ifstream stream(gfgFileName, std::ios_base::in | std::ios_base::binary);
	GFGFileReaderSTL stlFileReader(stream);
	GFGFileLoader gfgFile(&stlFileReader);
	gfgFile.ValidateAndOpen();

	// 
	assert(gfgFile.Header().animations.size() == 1);
	const GFGAnimationHeader anim = gfgFile.Header().animations[0];
	header = anim;
	assert(anim.quatType == GFGQuatLayout::WXYZ);		
	assert(anim.type == GFGAnimType::WITH_HIP_TRANSLATE);
	assert(anim.layout == GFGAnimationLayout::KEYS_OF_BONES);
	assert(anim.keyCount > 1);

	// Now Load
	uint32_t boneCount = gfgFile.Header().skeletons[anim.skeletonIndex].boneAmount;
	uint64_t dataPtr = 0;
	std::vector<uint8_t> data(gfgFile.AnimationKeyframeDataSize(0));
	gfgFile.AnimationKeyframeData(data.data(), 0);

	keyTimes.resize(anim.keyCount);
	std::memcpy(keyTimes.data(), data.data() + dataPtr,
				sizeof(float) * anim.keyCount);
	dataPtr += sizeof(float) * anim.keyCount;

	hipTranslations.resize(anim.keyCount);
	std::memcpy(hipTranslations.data(), data.data() + dataPtr,
				sizeof(float) * 3 * anim.keyCount);
	dataPtr += sizeof(float) * 3 * anim.keyCount;

	jointKeys.resize(boneCount);
	for(unsigned int i = 0; i < boneCount; i++)
	{
		jointKeys[i].resize(anim.keyCount);
		std::memcpy(jointKeys[i].data(), data.data() + dataPtr,
					sizeof(IEQuaternion) * anim.keyCount);
		dataPtr += sizeof(IEQuaternion) * anim.keyCount;
	}

	// Load Transforms
	bindPose = gfgFile.Header().bonetransformData.transforms;

	// Generate Joint Key Down Top Hierarchy
	bindPose.resize(boneCount);
	jointHierarchy.resize(boneCount);
	uint32_t index = 0;
	for(const GFGBone& bone : gfgFile.Header().skeletons[anim.skeletonIndex].bones)
	{
		bindPose[index] = gfgFile.Header().bonetransformData.transforms[bone.transformIndex];
		jointHierarchy[index] = bone.parentIndex;
		index++;
	}
	return GFGLoadError::OK;
}